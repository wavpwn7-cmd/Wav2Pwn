from common_batch_attack import main


if __name__ == "__main__":
    main(
        default_model_name="facebook/wav2vec2-conformer-rope-large-100h-ft",
        model_slug="wav2vec2_conformer_rope_large_100h_ft",
        paper_name="W2V2-Rope-Large-100h-ft",
    )
