"""The inherited 400-step formal queue is deliberately disabled in this repair checkout.

The unchanged v1 queue and its historical results remain in patr-reproduction.
V2 experiments require an explicit short config, selected method and run root.
No background queue, automatic retry or formal rollout is launched here.
"""


def main():
    raise SystemExit(
        'Formal queue disabled in v2. Select a reviewed short_validation configuration '
        '(1..40 steps) and run explicitly: python train.py --method METHOD '
        '--config configs/SHORT_CONFIG.json --run-root runs/SHORT_NAME. '
        'Use --resume only when that exact frozen run already has a checkpoint.'
    )


if __name__ == '__main__':
    main()
