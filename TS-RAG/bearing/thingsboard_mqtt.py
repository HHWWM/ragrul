"""
持续上传数据，通过 MQTT 发到 ThingsBoard。它不负责训练，也不负责复杂建模，主要负责“周期执行 + 发布 + 接收控制命令”。
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt

from bearing.bearing_inference_runtime import BearingRULRuntime
from bearing.utils import read_yaml


"""
    这个类做三件事:
    1. 定时调用在线推理
    2. 把结果发布到 ThingsBoard
    3. 响应远程控制指令，比如切换轴承目录、调整发布间隔
"""
class ThingsBoardRULPublisher:
    def __init__(self, config_path: str | Path, bearing_dir: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        self.cfg = read_yaml(self.config_path)
        self.tb = dict(self.cfg.get("thingsboard", {}))
        self.bearing_dir = str(Path(bearing_dir).expanduser().resolve())

        self.runtime = BearingRULRuntime(self.config_path)

        self.telemetry_topic = self.tb.get("telemetry_topic", self.tb.get("topic", "v1/devices/me/telemetry"))
        self.attributes_topic = self.tb.get("attributes_topic", "v1/devices/me/attributes")
        self.rpc_request_topic = self.tb.get("rpc_request_topic", "v1/devices/me/rpc/request/+")
        self.subscribe_shared_attributes = bool(self.tb.get("subscribe_shared_attributes", True))
        self.publish_interval_sec = int(self.tb.get("publish_interval_sec", 15))
        self.enabled = bool(self.tb.get("enabled", True))

        self.loop_windows = bool(self.tb.get("loop_windows", False))
        self.current_end_idx: Optional[int] = None
        self.min_end_idx: Optional[int] = None
        self.max_end_idx: Optional[int] = None

        self._lock = threading.Lock()
        self.client = self._build_client()

    def _env(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value.startswith("${") and value.endswith("}"):
            return os.getenv(value[2:-1], "")
        return value

    def _build_client(self) -> mqtt.Client:
        client_id = self._env(self.tb.get("client_id", "")) or ""
        protocol_name = str(self.tb.get("mqtt_version", "v311")).lower()
        protocol = mqtt.MQTTv5 if protocol_name in {"v5", "mqttv5"} else mqtt.MQTTv311

        client = mqtt.Client(client_id=client_id, protocol=protocol)

        access_token = self._env(self.tb.get("access_token", "")) or ""
        username = self._env(self.tb.get("username", "")) or access_token
        password = self._env(self.tb.get("password", "")) or None

        if username:
            client.username_pw_set(username=username, password=password)

        if bool(self.tb.get("use_tls", False)):
            ca_cert = self._env(self.tb.get("ca_cert", "")) or None
            certfile = self._env(self.tb.get("certfile", "")) or None
            keyfile = self._env(self.tb.get("keyfile", "")) or None
            insecure = bool(self.tb.get("tls_insecure", False))
            tls_version = ssl.PROTOCOL_TLS_CLIENT
            client.tls_set(
                ca_certs=ca_cert,
                certfile=certfile,
                keyfile=keyfile,
                tls_version=tls_version,
            )
            client.tls_insecure_set(insecure)

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.reconnect_delay_set(min_delay=1, max_delay=60)
        client.max_inflight_messages_set(int(self.tb.get("max_inflight_messages", 20)))
        client.max_queued_messages_set(int(self.tb.get("max_queued_messages", 0)))
        client.enable_logger()

        return client

    def _reset_window_state(self) -> None:
        info = self.runtime.get_valid_window_range(self.bearing_dir)
        self.min_end_idx = int(info["min_end_idx"])
        self.max_end_idx = int(info["max_end_idx"])

        initial_end_idx = self.tb.get("initial_end_idx", None)
        if initial_end_idx is None:
            self.current_end_idx = self.min_end_idx
        else:
            self.current_end_idx = max(self.min_end_idx, min(int(initial_end_idx), self.max_end_idx))

        print(
            f"[TB] window state initialized: "
            f"min_end_idx={self.min_end_idx}, max_end_idx={self.max_end_idx}, current_end_idx={self.current_end_idx}"
        )

    def _advance_window(self) -> None:
        if self.current_end_idx is None or self.min_end_idx is None or self.max_end_idx is None:
            self._reset_window_state()
            return

        if self.current_end_idx < self.max_end_idx:
            self.current_end_idx += 1
            return

        if self.loop_windows:
            self.current_end_idx = self.min_end_idx
        """
        连上之后订阅属性和 RPC 请求。
        """
    def _on_connect(self, client: mqtt.Client, userdata, flags, reason_code, properties=None):
        print(f"[TB] connected, reason_code={reason_code}")

        if self.subscribe_shared_attributes:
            client.subscribe(self.attributes_topic, qos=1)

        if bool(self.tb.get("subscribe_rpc", True)):
            client.subscribe(self.rpc_request_topic, qos=1)

        try:
            self._reset_window_state()
        except Exception as exc:
            print(f"[TB] window init error: {exc}")

        self.publish_attributes(
            {
                "service": "bearing_rul_runtime",
                "bearing_dir": self.bearing_dir,
                "model_seq_len": int(self.runtime.cfg["model"]["seq_len"]),
                "prediction_length": int(self.runtime.cfg["model"].get("prediction_length", 1)),
                "publish_interval_sec": self.publish_interval_sec,
                "loop_windows": self.loop_windows,
                "current_end_idx": self.current_end_idx,
                "min_end_idx": self.min_end_idx,
                "max_end_idx": self.max_end_idx,
            }
        )

    def _on_disconnect(self, client: mqtt.Client, userdata, disconnect_flags, reason_code, properties=None):
        print(f"[TB] disconnected, reason_code={reason_code}")

    def _apply_shared_attributes(self, payload: Dict[str, Any]) -> None:
        if "shared" in payload and isinstance(payload["shared"], dict):
            payload = payload["shared"]
        if "client" in payload and isinstance(payload["client"], dict):
            payload = payload["client"]

        need_reset = False

        with self._lock:
            if "publish_interval_sec" in payload:
                self.publish_interval_sec = max(1, int(payload["publish_interval_sec"]))

            if "enabled" in payload:
                self.enabled = bool(payload["enabled"])

            if "loop_windows" in payload:
                self.loop_windows = bool(payload["loop_windows"])

            if "bearing_dir" in payload and payload["bearing_dir"]:
                self.bearing_dir = str(Path(payload["bearing_dir"]).expanduser().resolve())
                need_reset = True

            if "current_end_idx" in payload:
                self.current_end_idx = int(payload["current_end_idx"])

        if need_reset:
            self._reset_window_state()

        print(f"[TB] shared attributes applied: {payload}")

    def _reply_rpc(self, request_id: str, payload: Dict[str, Any]) -> None:
        response_topic = f"v1/devices/me/rpc/response/{request_id}"
        self.client.publish(response_topic, json.dumps(payload, ensure_ascii=False), qos=1)

    def _handle_rpc(self, msg) -> None:
        request_id = msg.topic.rsplit("/", 1)[-1]
        body = json.loads(msg.payload.decode("utf-8")) if msg.payload else {}
        method = body.get("method")
        params = body.get("params", {})

        if method == "getStatus":
            self._reply_rpc(
                request_id,
                {
                    "enabled": self.enabled,
                    "bearing_dir": self.bearing_dir,
                    "publish_interval_sec": self.publish_interval_sec,
                    "loop_windows": self.loop_windows,
                    "current_end_idx": self.current_end_idx,
                    "min_end_idx": self.min_end_idx,
                    "max_end_idx": self.max_end_idx,
                },
            )
            return

        if method == "setEnabled":
            self.enabled = bool(params)
            self._reply_rpc(request_id, {"ok": True, "enabled": self.enabled})
            return

        if method == "setInterval":
            self.publish_interval_sec = max(1, int(params))
            self._reply_rpc(request_id, {"ok": True, "publish_interval_sec": self.publish_interval_sec})
            return

        if method == "setBearingDir":
            self.bearing_dir = str(Path(str(params)).expanduser().resolve())
            self._reset_window_state()
            self._reply_rpc(request_id, {"ok": True, "bearing_dir": self.bearing_dir})
            return

        if method == "setWindowEndIdx":
            self.current_end_idx = int(params)
            self._reply_rpc(request_id, {"ok": True, "current_end_idx": self.current_end_idx})
            return

        if method == "runInferenceOnce":
            result = self.runtime.predict(self.bearing_dir, end_idx=self.current_end_idx)
            self.publish_prediction(result)
            self._reply_rpc(request_id, {"ok": True, "result": result})
            return

        self._reply_rpc(request_id, {"ok": False, "error": f"unsupported rpc method: {method}"})
        """
        响应平台下发的 RPC。
        支持:
        getStatus
        setEnabled
        setInterval
        setBearingDir
        runInferenceOnce
        """
    def _on_message(self, client: mqtt.Client, userdata, msg) -> None:
        try:
            if msg.topic.startswith("v1/devices/me/rpc/request/"):
                self._handle_rpc(msg)
                return

            payload = json.loads(msg.payload.decode("utf-8")) if msg.payload else {}
            self._apply_shared_attributes(payload)
        except Exception as exc:
            print(f"[TB] on_message error: {exc}")

    def connect(self) -> None:
        host = self._env(self.tb.get("host", "localhost"))
        port = int(self._env(self.tb.get("port", 1883)))
        keepalive = int(self.tb.get("keepalive", 60))

        self.client.connect(host, port, keepalive=keepalive)
        self.client.loop_start()

    def publish_attributes(self, payload: Dict[str, Any]) -> None:
        info = self.client.publish(self.attributes_topic, json.dumps(payload, ensure_ascii=False), qos=1)
        info.wait_for_publish(timeout=10)

    def publish_prediction(self, result: Dict[str, Any]) -> None:
        ts = int(time.time() * 1000)

        telemetry = {
            "ts": ts,
            "values": {
                "bearing_id": result["bearing_id"],
                "pred_rul_norm": result["pred_rul_norm"],
                "pred_rul_steps_proxy": result["pred_rul_steps_proxy"],
                "topk_distance_mean": result["topk_distance_mean"],
                "topk_distance_min": result["topk_distance_min"],
                "topk_neighbor_rul_norm_mean": result["topk_neighbor_rul_norm_mean"],
                "topk_neighbor_rul_steps_mean": result["topk_neighbor_rul_steps_mean"],
                "window_start_idx": result["window_start_idx"],
                "window_end_idx": result["window_end_idx"],
                "window_size": result["window_size"],
                "total_files": result["total_files"],
            },
        }

        info = self.client.publish(self.telemetry_topic, json.dumps(telemetry, ensure_ascii=False), qos=1)
        info.wait_for_publish(timeout=10)

        self.publish_attributes(
            {
                "bearing_dir": result["bearing_dir"],
                "topk_indices": result["topk_indices"],
                "topk_neighbor_bearings": result["topk_neighbor_bearings"],
                "last_publish_ts": ts,
                "current_end_idx": result["window_end_idx"],
                "total_files": result["total_files"],
            }
        )

        print(json.dumps(telemetry, ensure_ascii=False))

    def publish_error(self, message: str) -> None:
        ts = int(time.time() * 1000)
        telemetry = {"ts": ts, "values": {"connector_error": message}}
        self.client.publish(self.telemetry_topic, json.dumps(telemetry, ensure_ascii=False), qos=1)
        print(f"[TB] error: {message}")
        """
        持续循环发布。
        """
    def run(self) -> None:
        self.connect()

        try:
            self._reset_window_state()

            while True:
                with self._lock:
                    enabled = self.enabled
                    bearing_dir = self.bearing_dir
                    interval = self.publish_interval_sec
                    current_end_idx = self.current_end_idx

                if enabled:
                    try:
                        result = self.runtime.predict(bearing_dir, end_idx=current_end_idx)
                        self.publish_prediction(result)

                        with self._lock:
                            self.current_end_idx = result["window_end_idx"]
                            self._advance_window()
                    except Exception as exc:
                        self.publish_error(str(exc))

                time.sleep(interval)

        finally:
            self.client.loop_stop()
            self.client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="ThingsBoard MQTT connector for bearing RUL")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bearing_dir", type=str, required=True)
    args = parser.parse_args()

    app = ThingsBoardRULPublisher(args.config, args.bearing_dir)
    app.run()


if __name__ == "__main__":
    main()