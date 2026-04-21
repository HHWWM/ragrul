from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from bearing.utils import read_yaml


def build_client(host: str, port: int, access_token: str) -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(access_token)
    client.connect(host, port, keepalive=60)
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description='Publish RUL telemetry to ThingsBoard')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--bearing_dir', type=str, required=True)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    tb = cfg['thingsboard']
    client = build_client(tb['host'], int(tb['port']), tb['access_token'])
    topic = tb.get('topic', 'v1/devices/me/telemetry')
    interval = int(tb.get('publish_interval_sec', 15))

    while True:
        proc = subprocess.run(
            ['python', '-m', 'bearing.infer_rul', '--config', args.config, '--bearing_dir', args.bearing_dir],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        payload = json.loads(proc.stdout)
        telemetry = {
            'ts': int(time.time() * 1000),
            'values': {
                'bearing_dir': payload['bearing_dir'],
                'pred_rul_norm': payload['pred_rul_norm'],
                'topk_distance_mean': payload['topk_distance_mean'],
            },
        }
        client.publish(topic, json.dumps(telemetry), qos=1)
        client.loop(timeout=1.0)
        print(json.dumps(telemetry, ensure_ascii=False))
        time.sleep(interval)


if __name__ == '__main__':
    main()
