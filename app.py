from __future__ import annotations

import os
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

from compression_lib import ensure_directories, run_compression_pipeline
from metrics import evaluate_metrics


st.set_page_config(
    page_title="Medical Image Compression System",
    layout="wide",
)


def create_metric_table(results: dict) -> list[dict]:
    table = []
    for method_name, result in results.items():
        metric_values = evaluate_metrics(
            original=result.original_image,
            reconstructed=result.reconstructed_image,
            compressed_payload=result.compressed_payload,
        )
        table.append(
            {
                "Method": method_name,
                "Compression Ratio": round(metric_values["compression_ratio"], 4),
                "MSE": round(metric_values["mse"], 4),
                "PSNR": round(metric_values["psnr"], 4) if metric_values["psnr"] != float("inf") else "inf",
                "SSIM": round(metric_values["ssim"], 4),
            }
        )
    return table


def plot_images(results: dict) -> None:
    for method_name, result in results.items():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Original - {method_name}")
            # Convert grayscale to RGB for proper display
            original_rgb = np.stack([result.original_image] * 3, axis=-1) if result.original_image.ndim == 2 else result.original_image
            st.image(original_rgb, use_container_width=True)
        with col2:
            st.subheader(f"Reconstructed - {method_name}")
            # Convert grayscale to RGB for proper display
            reconstructed_rgb = np.stack([result.reconstructed_image] * 3, axis=-1) if result.reconstructed_image.ndim == 2 else result.reconstructed_image
            st.image(reconstructed_rgb
            st.subheader(f"Reconstructed - {method_name}")
            # Convert grayscale to RGB for proper display
            reconstructed_rgb = np.stack([result.reconstructed_image] * 3, axis=-1) if result.reconstructed_image.ndim == 2 else result.reconstructed_image
            st.image(reconstructed_rgb, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(result.original_image.ravel(), bins=32, alpha=0.5, label="Original")
        ax.hist(result.reconstructed_image.ravel(), bins=32, alpha=0.5, label="Reconstructed")
        ax.set_title(f"Histogram Comparison - {method_name}")
        ax.legend()
        st.pyplot(fig)


def main() -> None:
    ensure_directories()

    st.title("Medical Image Compression System")
    st.write(
        "Compress X-ray, MRI, CT, or DICOM images using wavelet-based and JPEG-LS-inspired methods, "
        "then compare distortion metrics."
    )

    uploaded_file = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg", "dcm"])

    method = st.selectbox("Select compression method", ["both", "wavelet", "jpegls"])
    resize_enabled = st.checkbox("Resize image before compression", value=False)

    width = st.number_input("Resize width", min_value=64, max_value=2048, value=512, step=32)
    height = st.number_input("Resize height", min_value=64, max_value=2048, value=512, step=32)

    st.markdown("### Wavelet Parameters")
    wavelet = st.selectbox("Wavelet family", ["haar", "db1", "db2", "coif1", "sym2"], index=0)
    level = st.slider("Wavelet decomposition level", min_value=1, max_value=4, value=2)
    threshold = st.slider("Wavelet threshold ratio", min_value=0.01, max_value=0.50, value=0.08, step=0.01)

    st.markdown("### JPEG-LS-Inspired Parameters")
    near = st.slider("Near-lossless value", min_value=0, max_value=10, value=0, step=1)

    if st.button("Compress and Analyze", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload an image first.")
            return

        suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        resize_to = (int(width), int(height)) if resize_enabled else None

        results = run_compression_pipeline(
            image_path=temp_path,
            method=method,
            resize_to=resize_to,
            wavelet=wavelet,
            level=int(level),
            threshold_ratio=float(threshold),
            near=int(near),
        )

        st.success("Compression completed successfully.")
        st.dataframe(create_metric_table(results), use_container_width=True)
        plot_images(results)

        comparison_note = []
        for method_name, result in results.items():
            metric_values = evaluate_metrics(
                original=result.original_image,
                reconstructed=result.reconstructed_image,
                compressed_payload=result.compressed_payload,
            )
            comparison_note.append(
                f"{method_name}: CR={metric_values['compression_ratio']:.3f}, "
                f"PSNR={metric_values['psnr']:.3f}, SSIM={metric_values['ssim']:.3f}"
            )

        st.markdown("### Interpretation")
        st.write(
            "Higher compression ratio means stronger size reduction. Higher PSNR and SSIM mean better quality. "
            "Try changing threshold and near-lossless values to observe the trade-off."
        )
        st.write(" | ".join(comparison_note))


if __name__ == "__main__":
    main()
