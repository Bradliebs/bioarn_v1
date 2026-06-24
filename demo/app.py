"""Bio-ARN 2.0 Interactive Demo

Launch: python demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from demo.models import (
    DOCS_URL,
    PAPER_URL,
    create_live_learning_session,
    format_token_html,
    format_top_matches,
    get_demo_overview_stats,
    get_energy_dashboard_data,
    get_mnist_model,
    get_multimodal_model,
    get_text_model,
    image_tensor_to_array,
)
from demo.visualizations import (
    ccc_activation_map,
    confidence_bar_chart,
    energy_comparison_chart,
    spike_raster_plot,
)

APP_LIVE_SESSION = create_live_learning_session()


def _pick_image(drawn_image, uploaded_image):
    if isinstance(drawn_image, dict):
        if drawn_image.get("composite") is not None or drawn_image.get("background") is not None:
            return drawn_image
        if any(layer is not None for layer in drawn_image.get("layers", [])):
            return drawn_image
    elif drawn_image is not None:
        return drawn_image
    return uploaded_image


def _header_markdown() -> str:
    stats = get_demo_overview_stats()
    return f"""
```text
██████╗ ██╗ ██████╗        █████╗ ██████╗ ███╗   ██╗    ██████╗      ██████╗
██╔══██╗██║██╔═══██╗      ██╔══██╗██╔══██╗████╗  ██║    ╚════██╗    ██╔═████╗
██████╔╝██║██║   ██║█████╗███████║██████╔╝██╔██╗ ██║     █████╔╝    ██║██╔██║
██╔══██╗██║██║   ██║╚════╝██╔══██║██╔══██╗██║╚██╗██║    ██╔═══╝     ████╔╝██║
██████╔╝██║╚██████╔╝      ██║  ██║██║  ██║██║ ╚████║    ███████╗    ╚██████╔╝
╚═════╝ ╚═╝ ╚═════╝       ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚══════╝     ╚═════╝
```

Brain-inspired spiking cognition with honest abstention, continual learning, sparse memory,
and neuromorphic efficiency.

- **Docs:** [Architecture & API]({DOCS_URL})
- **Paper / notes:** [BioARN_Architecture.md]({PAPER_URL})
- **Current model stats:** {int(stats["cccs_committed"])} committed CCCs · {float(stats["memory_mb"]):.2f} MB estimated memory
"""


def _stats_markdown(active_cccs: int, sparsity_pct: float, status: str) -> str:
    return (
        f"**Margin gate:** {status}  \n"
        f"**Active CCCs:** {active_cccs}  \n"
        f"**Sparsity:** {sparsity_pct:.1f}%"
    )


def _digit_demo(drawn_image, uploaded_image):
    image = _pick_image(drawn_image, uploaded_image)
    result = get_mnist_model().classify(image)
    return (
        result.label_text,
        confidence_bar_chart(result.class_scores),
        _stats_markdown(result.active_cccs, result.sparsity_pct, result.margin_status),
        spike_raster_plot(result.input_spikes),
        ccc_activation_map(result.ccc_activations),
    )


def _text_demo(prompt: str, max_tokens: int, temperature: float, method: str):
    result = get_text_model().generate_text(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        method=method,
    )
    metrics = (
        f"**Tokens/sec:** {result.tokens_per_sec:.1f}  \n"
        f"**Concepts activated:** {result.concepts_activated}  \n"
        f"**SDM retrievals:** {result.sdm_retrievals}"
    )
    return result.generated_text, format_token_html(result.prompt, result.generated_tokens), metrics


def _toggle_cross_mode(mode: str):
    show_image = mode == "Image→Text"
    show_text = mode == "Text→Image"
    return gr.update(visible=show_image), gr.update(visible=show_text)


def _cross_modal_demo(mode: str, image_input, text_input: str):
    model = get_multimodal_model()
    if mode == "Image→Text":
        result = model.retrieve(mode="image-to-text", image=_pick_image(image_input, None))
        return (
            result.retrieved_text or "unknown",
            None,
            f"**Association strength:** {result.association_strength:.3f}",
            format_top_matches(result.top_matches),
        )

    result = model.retrieve(mode="text-to-image", text=text_input)
    image = image_tensor_to_array(result.retrieved_image) if result.retrieved_image is not None else None
    return (
        result.retrieved_text or text_input,
        image,
        f"**Association strength:** {result.association_strength:.3f}",
        format_top_matches(result.top_matches),
    )


def _teach_live_pattern(image_input, label: str):
    outcome = APP_LIVE_SESSION.teach(_pick_image(image_input, None), label)
    retention_lines = "\n".join(
        f"- **{name}**: {'kept ✅' if kept else 'missed ❌'}"
        for name, kept in sorted(outcome.retention.items())
    )
    return (
        outcome.message,
        (
            f"**Total CCCs:** {outcome.pool_stats['total']}  \n"
            f"**Committed:** {outcome.pool_stats['committed']}  \n"
            f"**Available:** {outcome.pool_stats['available']}"
        ),
        f"**Immediate recognition:** {outcome.recognized_label}",
        retention_lines,
    )


def _recognize_live_pattern(image_input):
    return APP_LIVE_SESSION.recognize(_pick_image(image_input, None))


def create_app(*, load_models: bool = True) -> gr.Blocks:
    if load_models:
        get_mnist_model()
        get_text_model()
        get_multimodal_model()
        get_energy_dashboard_data()

    dashboard = get_energy_dashboard_data() if load_models else None
    with gr.Blocks(title="Bio-ARN 2.0 Interactive Demo") as demo:
        gr.Markdown(_header_markdown())

        with gr.Tabs():
            with gr.Tab("🔢 Digit Recognition"):
                with gr.Row():
                    with gr.Column():
                        drawn_image = gr.Sketchpad(
                            image_mode="L",
                            type="numpy",
                            label="Draw a digit",
                            height=280,
                            width=280,
                        )
                        uploaded_image = gr.Image(
                            image_mode="L",
                            type="numpy",
                            label="...or upload a 28×28 grayscale image",
                            height=280,
                        )
                        gr.Examples(
                            examples=[
                                [image_tensor_to_array(get_mnist_model().example_digits[digit])]
                                for digit in (0, 3, 7)
                            ],
                            inputs=uploaded_image,
                            label="Quick examples",
                        )
                        recognize_button = gr.Button("Recognize digit", variant="primary")
                    with gr.Column():
                        digit_prediction = gr.Textbox(label="Prediction")
                        digit_chart = gr.Plot(label="Confidence by class")
                        digit_stats = gr.Markdown()
                        with gr.Accordion("Internal activity", open=False):
                            digit_raster = gr.Plot(label="Spike raster")
                            digit_activation = gr.Plot(label="CCC activation map")

                recognize_button.click(
                    _digit_demo,
                    inputs=[drawn_image, uploaded_image],
                    outputs=[
                        digit_prediction,
                        digit_chart,
                        digit_stats,
                        digit_raster,
                        digit_activation,
                    ],
                )

            with gr.Tab("📝 Text Generation"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="Prompt",
                            value="The town",
                            lines=4,
                            placeholder="Type a prompt for Bio-ARN to continue...",
                        )
                        with gr.Row():
                            max_tokens = gr.Slider(10, 200, value=60, step=1, label="Max tokens")
                            temperature = gr.Slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
                        method = gr.Dropdown(
                            choices=["greedy", "beam", "top-k", "top-p"],
                            value="beam",
                            label="Decoding method",
                        )
                        gr.Examples(
                            examples=[
                                ["The town"],
                                ["Once upon a time"],
                                ["Rain on the roof"],
                            ],
                            inputs=prompt,
                            label="Prompt examples",
                        )
                        text_button = gr.Button("Generate continuation", variant="primary")
                    with gr.Column():
                        generated_text = gr.Textbox(label="Generated text", lines=6)
                        token_html = gr.HTML(label="Token-by-token view")
                        generation_metrics = gr.Markdown()

                text_button.click(
                    _text_demo,
                    inputs=[prompt, max_tokens, temperature, method],
                    outputs=[generated_text, token_html, generation_metrics],
                )

            with gr.Tab("🔗 Cross-Modal"):
                with gr.Row():
                    with gr.Column():
                        cross_mode = gr.Radio(
                            choices=["Image→Text", "Text→Image"],
                            value="Image→Text",
                            label="Mode",
                        )
                        cross_image = gr.Sketchpad(
                            image_mode="L",
                            type="numpy",
                            label="Draw or paste a simple pattern",
                            height=280,
                            width=280,
                            visible=True,
                        )
                        cross_text = gr.Textbox(
                            label="Type a label",
                            value="horizontal",
                            visible=False,
                        )
                        gr.Examples(
                            examples=[
                                [image_tensor_to_array(get_multimodal_model().patterns["horizontal"])],
                                [image_tensor_to_array(get_multimodal_model().patterns["box"])],
                                [image_tensor_to_array(get_multimodal_model().patterns["checker"])],
                            ],
                            inputs=cross_image,
                            label="Pattern examples",
                        )
                        gr.Examples(
                            examples=[["horizontal"], ["diagonal"], ["frame"]],
                            inputs=cross_text,
                            label="Label examples",
                        )
                        cross_button = gr.Button("Retrieve association", variant="primary")
                    with gr.Column():
                        cross_result_text = gr.Textbox(label="Retrieved label / query")
                        cross_result_image = gr.Image(label="Retrieved visual pattern", image_mode="L")
                        cross_strength = gr.Markdown()
                        cross_top_matches = gr.Markdown()

                cross_mode.change(
                    _toggle_cross_mode,
                    inputs=[cross_mode],
                    outputs=[cross_image, cross_text],
                )
                cross_button.click(
                    _cross_modal_demo,
                    inputs=[cross_mode, cross_image, cross_text],
                    outputs=[
                        cross_result_text,
                        cross_result_image,
                        cross_strength,
                        cross_top_matches,
                    ],
                )

            with gr.Tab("🧠 Live Learning"):
                with gr.Row():
                    with gr.Column():
                        teach_image = gr.Sketchpad(
                            image_mode="L",
                            type="numpy",
                            label="Draw a new pattern",
                            height=280,
                            width=280,
                        )
                        teach_label = gr.Textbox(
                            label="Name this pattern",
                            value="zigzag",
                            placeholder="e.g. zigzag",
                        )
                        teach_button = gr.Button("Teach Bio-ARN", variant="primary")
                        recognize_live_button = gr.Button("Recognize current pattern")
                    with gr.Column():
                        teach_message = gr.Markdown()
                        teach_pool = gr.Markdown()
                        teach_recognition = gr.Markdown()
                        forgetting_report = gr.Markdown()

                teach_button.click(
                    _teach_live_pattern,
                    inputs=[teach_image, teach_label],
                    outputs=[teach_message, teach_pool, teach_recognition, forgetting_report],
                )
                recognize_live_button.click(
                    _recognize_live_pattern,
                    inputs=[teach_image],
                    outputs=[teach_recognition],
                )

            with gr.Tab("⚡ Energy Dashboard"):
                energy_plot = gr.Plot(value=energy_comparison_chart(), label="Energy per inference")
                energy_callout = gr.Markdown(
                    value=(
                        f"## {dashboard.efficiency_callout_x:.0f}× efficiency\n"
                        "Matched-transformer benchmark: projected Loihi deployment uses dramatically less "
                        "energy than an A100-class transformer baseline."
                    )
                    if dashboard is not None
                    else "Energy dashboard loading..."
                )
                energy_stats = gr.Markdown(
                    value=(
                        f"**Estimated edge battery life:** {dashboard.battery_life_hours:.1f} hours at 1k inf/sec  \n"
                        f"**Total neurons:** {dashboard.architecture_stats['total_neurons']:,}  \n"
                        f"**Total synapses:** {dashboard.architecture_stats['total_synapses']:,}  \n"
                        f"**Active CCCs:** {dashboard.architecture_stats['active_cccs']}  \n"
                        f"**Estimated mapped memory:** {dashboard.architecture_stats['memory_mb']} MB"
                    )
                    if dashboard is not None
                    else "Architecture stats loading..."
                )
                gr.Markdown(
                    value=(
                        f"| Backend | Energy / inference |\n| --- | ---: |\n"
                        + "\n".join(
                            f"| {name} | {value * 1e6:.2f} µJ |"
                            for name, value in dashboard.energies_joules.items()
                        )
                    )
                    if dashboard is not None
                    else ""
                )

    return demo


if __name__ == "__main__":
    create_app().launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"))
