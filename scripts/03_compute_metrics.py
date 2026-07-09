from ru_bench.metrics_asr import compute_asr_metrics, save_metrics


def main() -> None:
    metrics = compute_asr_metrics()
    save_metrics(metrics)
    print(f"n={metrics['n']} cer={metrics['cer']:.3f} wer={metrics['wer']:.3f}")


if __name__ == "__main__":
    main()
