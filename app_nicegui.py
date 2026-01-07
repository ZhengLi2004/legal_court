"""Creates a web-based graphical user interface (GUI) for the Legal MAS.

This module uses the NiceGUI library to build an interactive web front end for
the debate simulation. The UI allows users to:
-   Select a legal case from a predefined list.
-   Initialize the `DebateEngine` with the chosen case.
-   Step through the debate turn by turn.
-   View a real-time, interactive visualization of the debate graph using ECharts.
-   Read the full, narrated transcript of the debate.
-   View the final judgment in a pop-up dialog once the debate concludes.
-   Reset the system to its initial state.

The application's state is managed by the `AppState` class to ensure data
persistence across user interactions.
"""

from nicegui import ui

from data.loader import CaseDataLoader
from mas.config import SystemConfig
from mas.engine import DebateEngine
from vis.adapter import EChartsAdapter


class LogView(ui.scroll_area):
    """A custom NiceGUI component for displaying a scrollable, auto-scrolling log.

    This class extends `ui.scroll_area` to create a styled container that is
    ideal for showing transcripts or log messages. It automatically scrolls to the
    bottom whenever a new line is pushed, ensuring the latest content is always
    visible.

    Attributes:
        lines_container: A `ui.column` within the scroll area that holds the
            individual log labels.
    """

    def __init__(self, **kwargs):
        """Initialize the LogView component."""
        super().__init__(**kwargs)
        self.classes("w-full h-full border rounded bg-gray-50 p-2")
        self.lines_container = ui.column().classes("w-full gap-1")

        with self:
            self.lines_container.move(self)

    def push(self, line: str):
        """Add a new line of text to the log view.

        Args:
            line: The string message to add.
        """
        with self.lines_container:
            ui.label(line).classes(
                "text-sm font-mono break-words whitespace-pre-wrap text-gray-800"
            )

        self.scroll_to(percent=1.0)

    def clear(self):
        """Remove all lines from the log view."""
        self.lines_container.clear()


class AppState:
    """Manage the global state of the web application.

    This class holds all the critical state objects for the application, such
    as the debate engine instance, the case data loader, and the list of
    available samples. Using a single state class helps in managing the
    application's data flow and persistence across UI interactions.

    Attributes:
        engine: The instance of the `DebateEngine` that runs the simulation.
        loader: The `CaseDataLoader` instance to load case data from files.
        samples: A list of `CaseData` objects available for simulation.
        selected_index: The index of the currently selected case in the `samples` list.
        is_initialized: A boolean flag indicating if the engine has been set up.
    """

    def __init__(self):
        """Initialize the application state."""
        self.engine = DebateEngine(config=SystemConfig(), judge_config={})
        self.loader = CaseDataLoader("data/sampling")
        self.samples = self.loader.load_all(limit=20)
        self.selected_index = 0 if self.samples else None
        self.is_initialized = False

    @property
    def current_case(self):
        """The currently selected `CaseData` object."""
        if self.samples and self.selected_index is not None:
            if 0 <= self.selected_index < len(self.samples):
                return self.samples[self.selected_index]

        return None


state = AppState()
chart_view = None
log_view = None
stats_label = None
verdict_dialog = None
verdict_content = None


async def on_init_click():
    """Async event handler for the 'Initialize' button.

    This function retrieves the currently selected case, displays a loading
    dialog, and calls the `DebateEngine.setup()` method to prepare the
    simulation. It then updates the UI to reflect the initial state.
    """
    case = state.current_case

    if not case:
        ui.notify("请先选择一个案件", type="warning")
        return

    with ui.dialog() as loading_dialog, ui.card():
        ui.label("正在初始化案件...").classes("text-xl")
        ui.spinner(size="lg")

    loading_dialog.open()

    try:
        ui.notify(f"正在初始化案件: {case.title}...")
        case_dict = case.model_dump()
        await state.engine.setup(case_data=case_dict, verbose=True)
        state.is_initialized = True
        update_ui()
        ui.notify("系统初始化完成！", type="positive")

    finally:
        loading_dialog.close()


async def on_next_turn_click():
    """Async event handler for the 'Next Turn' button.

    This function advances the debate by one step by calling `DebateEngine.step()`.
    It shows a loading dialog during processing. After the step is complete,
    it updates the UI. If the debate has finished, it opens the verdict dialog.
    """
    if not state.is_initialized:
        ui.notify("请先初始化系统", type="warning")
        return

    if state.engine.is_finished:
        ui.notify("辩论已结束", type="info")
        return

    turn_name = state.engine.current_turn.value.capitalize()

    with ui.dialog() as step_dialog, ui.card():
        ui.label(f"正在运行 {turn_name} 回合...").classes("text-xl")
        ui.spinner(size="lg")

    step_dialog.open()

    try:
        await state.engine.step()
        update_ui()

        if state.engine.is_finished:
            ui.notify("辩论结束，判决已生成！", type="positive")

            if verdict_dialog:
                verdict_dialog.open()

    finally:
        step_dialog.close()


def on_reset_click():
    """Event handler for the 'Reset' button.

    This function resets the application to its initial state by creating a new
    `DebateEngine` instance and clearing all UI components (graph, logs, stats).
    """
    state.engine = DebateEngine(config=SystemConfig(), judge_config={})
    state.is_initialized = False

    if chart_view:
        chart_view.options.clear()
        chart_view.update()

    if log_view:
        log_view.clear()

    update_stats()
    ui.notify("系统已重置", type="warning")


def update_ui():
    """Update all major UI components based on the current engine state.

    This function is the central point for refreshing the user interface. It
    re-parses the debate graph for the ECharts view, repopulates the log view
    with the latest transcript, and updates the verdict dialog content.
    """
    if state.engine.graph and chart_view:
        option = EChartsAdapter.parse_graph(state.engine.graph)

        option["tooltip"]["formatter"] = (
            ":(params) => { var c = params.data.full_content ? params.data.full_content : params.name; return c.replace(/(.{50})/g, '$1<br/>'); }"
        )

        chart_view.options.clear()
        chart_view.options.update(option)
        chart_view.update()

    if log_view:
        log_view.clear()

        for line in state.engine.transcript:
            log_view.push(line)
            log_view.push("--------------------------------------------------")

    update_stats()

    if state.engine.is_finished and state.engine.judgment_document:
        if verdict_content:
            verdict_content.content = state.engine.judgment_document.replace(
                "\n", "<br>"
            )


def update_stats():
    """Update the statistics label with the latest round, turn, and convergence info."""
    if stats_label:
        round_idx = state.engine.round_idx
        snapshot = state.engine.get_snapshot()
        last_log = snapshot.get("last_log", {}) if snapshot else {}
        conv = last_log.get("convergence", {}).get("sma", 0.0) if last_log else 0.0
        turn = state.engine.current_turn.value if state.engine.current_turn else "Ready"

        stats_label.text = (
            f"Round: {round_idx} | Turn: {turn} | Convergence: {conv:.4f}"
        )


@ui.page("/")
def index():
    """Define the main page layout and initializes all UI components.

    This function is decorated with `@ui.page('/')` and is called by NiceGUI to
    build the user interface. It constructs the header, the main two-column
    layout (graph and transcript), the control panel, and the verdict dialog.
    """
    global chart_view, log_view, stats_label, verdict_dialog, verdict_content

    with ui.header().classes("bg-blue-700 items-center justify-between"):
        ui.label("⚖️ Legal MAS Console (NiceGUI)").classes("text-xl font-bold")

        with ui.row():
            ui.button("Reset", on_click=on_reset_click, color="red").props("flat")

    with ui.dialog() as verdict_dialog, ui.card().classes("w-3/4 h-3/4"):
        ui.label("🏛️ Civil Judgment").classes("text-2xl font-serif text-center w-full")
        ui.separator()

        with ui.scroll_area().classes("h-full p-4"):
            verdict_content = ui.html("No verdict yet.", sanitize=False).classes(
                "font-serif text-lg leading-loose"
            )

        ui.button("Close", on_click=verdict_dialog.close)

    with ui.row().classes("w-full h-[calc(100vh-60px)] no-wrap"):
        with ui.column().classes("w-3/5 h-full p-2"):
            with ui.card().classes("w-full p-2 mb-2"):
                with ui.row().classes("items-center w-full gap-4"):
                    options = {
                        i: f"{c.uid[:6]} - {c.title[:15]}..."
                        for i, c in enumerate(state.samples)
                    }

                    ui.select(
                        options, value=state.selected_index, label="Select Case"
                    ).bind_value(state, "selected_index").classes("w-64")

                    ui.button("Initialize", on_click=on_init_click, icon="rocket")

                    ui.button(
                        "Next Turn", on_click=on_next_turn_click, icon="play_arrow"
                    ).props("color=green")

                    stats_label = ui.label("Ready").classes(
                        "text-gray-600 font-mono ml-auto"
                    )

            chart_view = ui.echart({}).classes(
                "w-full flex-grow border rounded-lg shadow-sm bg-white"
            )

        with ui.column().classes("w-2/5 h-full p-2"):
            ui.label("📝 Debate Transcript").classes("text-lg font-bold mb-2")
            log_view = LogView()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Legal MAS", port=8080, reload=True)
