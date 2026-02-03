"""NiceGUI frontend for album organization."""

import asyncio
from pathlib import Path

from nicegui import ui, app

from app.nicegui.api_client import api_client
from app.nicegui.state.app_state import app_state


DEFAULT_STEPS = [
    "discover_images",
    "score_iqa",
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "select_best"
]


async def refresh_albums():
    """Refresh album list."""
    albums = await api_client.list_albums()
    return albums


def create_header():
    """Create the app header."""
    with ui.header().classes("bg-blue-600 text-white"):
        ui.label("Album Organizer").classes("text-xl font-bold")
        ui.space()
        with ui.row():
            ui.button("Home", on_click=lambda: ui.navigate.to("/"))
            ui.button("Results", on_click=lambda: ui.navigate.to("/results"))
            ui.button("People", on_click=lambda: ui.navigate.to("/people"))


@ui.page("/")
async def home_page():
    """Home page - album selection and pipeline configuration."""
    create_header()

    with ui.column().classes("w-full max-w-4xl mx-auto p-4"):
        ui.label("Album Organization").classes("text-2xl font-bold mb-4")

        # Album selection section
        with ui.card().classes("w-full mb-4"):
            ui.label("Select or Create Album").classes("text-lg font-semibold mb-2")

            albums = await refresh_albums()

            if albums:
                album_options = {a["id"]: f"{a['name']} ({a['image_count']} images)" for a in albums}
                album_select = ui.select(
                    options=album_options,
                    label="Select Album",
                    on_change=lambda e: setattr(app_state, "current_album",
                        next((a for a in albums if a["id"] == e.value), None))
                ).classes("w-full")
            else:
                ui.label("No albums found. Create one below.")

            ui.separator()

            with ui.row().classes("w-full gap-4"):
                name_input = ui.input(label="Album Name").classes("flex-1")
                path_input = ui.input(label="Source Path").classes("flex-1")

            async def create_album():
                if name_input.value and path_input.value:
                    album = await api_client.create_album(name_input.value, path_input.value)
                    app_state.current_album = album
                    ui.notify(f"Created album: {album['name']}")
                    ui.navigate.to("/")

            ui.button("Create Album", on_click=create_album).classes("mt-2")

        # Pipeline configuration section
        with ui.card().classes("w-full mb-4"):
            ui.label("Pipeline Configuration").classes("text-lg font-semibold mb-2")

            steps = await api_client.list_steps()

            with ui.column().classes("w-full"):
                for step in steps:
                    is_default = step["name"] in DEFAULT_STEPS

                    with ui.row().classes("items-center"):
                        cb = ui.checkbox(
                            step["display_name"],
                            value=is_default,
                            on_change=lambda e, s=step["name"]: (
                                app_state.selected_steps.append(s) if e.value
                                else app_state.selected_steps.remove(s) if s in app_state.selected_steps
                                else None
                            )
                        )
                        ui.label(f"({step['category']})").classes("text-gray-500 text-sm")

                    if is_default:
                        app_state.selected_steps.append(step["name"])

        # Run button
        progress_label = ui.label("").classes("mt-2")
        progress_bar = ui.linear_progress(value=0).classes("w-full mt-2")
        progress_bar.visible = False

        async def run_pipeline():
            if not app_state.current_album:
                ui.notify("Please select an album first", type="warning")
                return

            progress_bar.visible = True
            progress_label.text = "Starting pipeline..."

            job_id = await api_client.run_pipeline(
                album_id=app_state.current_album["id"],
                steps=app_state.selected_steps or None
            )

            app_state.current_job_id = job_id

            def on_progress(data):
                app_state.progress = data["progress"]
                app_state.current_step = data["step"]
                app_state.progress_message = data.get("message", "")
                progress_bar.value = data["progress"]
                progress_label.text = f"{data['step']}: {data.get('message', '')}"

            def on_complete():
                ui.notify("Pipeline completed!", type="positive")
                progress_label.text = "Complete!"
                ui.navigate.to(f"/results?job_id={job_id}")

            def on_error(message):
                ui.notify(f"Pipeline failed: {message}", type="negative")
                progress_label.text = f"Error: {message}"

            await api_client.subscribe_progress(job_id, on_progress, on_complete, on_error)

        ui.button("Run Pipeline", on_click=run_pipeline).classes("mt-4").props("color=primary")


@ui.page("/results")
async def results_page():
    """Results page - display pipeline results."""
    create_header()

    with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
        ui.label("Pipeline Results").classes("text-2xl font-bold mb-4")

        job_id = app.storage.browser.get("job_id") or app_state.current_job_id

        if not job_id:
            ui.label("No results to display. Run a pipeline first.")
            return

        result = await api_client.get_pipeline_result(job_id)

        # Summary stats
        with ui.card().classes("w-full mb-4"):
            ui.label("Summary").classes("text-lg font-semibold mb-2")

            with ui.row().classes("gap-8"):
                with ui.column():
                    ui.label(f"Total Images: {result['total_images']}")
                    ui.label(f"After Filtering: {result['filtered_images']}")

                with ui.column():
                    ui.label(f"Clusters: {result['num_clusters']}")
                    ui.label(f"Selected: {result['num_selected']}")

                with ui.column():
                    ui.label(f"Duration: {result['total_duration_ms']}ms")

        # Selected images
        with ui.card().classes("w-full mb-4"):
            ui.label("Selected Images").classes("text-lg font-semibold mb-2")

            with ui.row().classes("flex-wrap gap-2"):
                for image_path in result["selected_images"][:20]:
                    path = Path(image_path)
                    if path.exists():
                        ui.image(str(path)).classes("w-32 h-32 object-cover rounded")
                    else:
                        ui.label(path.name).classes("text-sm")

            if len(result["selected_images"]) > 20:
                ui.label(f"... and {len(result['selected_images']) - 20} more")

        # Clusters
        with ui.card().classes("w-full"):
            ui.label("Clusters").classes("text-lg font-semibold mb-2")

            for cluster_id, images in sorted(result["scene_clusters"].items(), key=lambda x: int(x[0])):
                cluster_name = "Noise" if int(cluster_id) == -1 else f"Cluster {cluster_id}"

                with ui.expansion(f"{cluster_name} ({len(images)} images)").classes("w-full"):
                    with ui.row().classes("flex-wrap gap-2"):
                        for image_path in images[:10]:
                            path = Path(image_path)
                            if path.exists():
                                ui.image(str(path)).classes("w-24 h-24 object-cover rounded")

                        if len(images) > 10:
                            ui.label(f"+{len(images) - 10} more").classes("text-gray-500")

        # Export section
        with ui.card().classes("w-full mt-4"):
            ui.label("Export Results").classes("text-lg font-semibold mb-2")

            export_path = ui.input(label="Output Directory").classes("w-full")

            with ui.row().classes("gap-4 mt-2"):
                include_selected = ui.checkbox("Include Selected", value=True)
                include_all = ui.checkbox("Include All Filtered", value=False)
                organize_cluster = ui.checkbox("Organize by Cluster", value=False)
                organize_person = ui.checkbox("Organize by Person", value=False)

            with ui.row().classes("gap-4 mt-2"):
                copy_mode = ui.toggle(["Copy", "Symlink"], value="Copy")

            async def do_export():
                if not export_path.value:
                    ui.notify("Please enter an output directory", type="warning")
                    return

                result = await api_client.export_result(
                    job_id=job_id,
                    output_path=export_path.value,
                    include_selected=include_selected.value,
                    include_all_filtered=include_all.value,
                    organize_by_cluster=organize_cluster.value,
                    organize_by_person=organize_person.value,
                    copy_mode="symlink" if copy_mode.value == "Symlink" else "copy"
                )

                if result["success"]:
                    ui.notify(f"Exported {result['files_exported']} files!", type="positive")
                else:
                    ui.notify(f"Export had errors: {result['errors']}", type="warning")

            ui.button("Export", on_click=do_export).props("color=primary")


@ui.page("/people")
async def people_page():
    """People page - view and manage detected people."""
    create_header()

    with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
        ui.label("Detected People").classes("text-2xl font-bold mb-4")

        # Album selection
        albums = await refresh_albums()
        if not albums:
            ui.label("No albums found. Create one from the Home page.")
            return

        album_options = {a["id"]: f"{a['name']} ({a['image_count']} images)" for a in albums}

        selected_album_id = ui.select(
            options=album_options,
            label="Select Album",
            value=app_state.current_album["id"] if app_state.current_album else None
        ).classes("w-full max-w-md mb-4")

        people_container = ui.column().classes("w-full")

        async def load_people():
            people_container.clear()
            if not selected_album_id.value:
                return

            people = await api_client.list_people(selected_album_id.value)

            if not people:
                with people_container:
                    ui.label("No people detected. Run a pipeline with face clustering first.")
                return

            with people_container:
                # Merge selection
                merge_selection = []

                with ui.row().classes("w-full mb-4 items-center"):
                    ui.label(f"Found {len(people)} people").classes("text-lg")
                    ui.space()

                    async def do_merge():
                        if len(merge_selection) < 2:
                            ui.notify("Select at least 2 people to merge", type="warning")
                            return
                        await api_client.merge_people(selected_album_id.value, merge_selection)
                        ui.notify("People merged!", type="positive")
                        merge_selection.clear()
                        await load_people()

                    ui.button("Merge Selected", on_click=do_merge).props("outline")

                # People grid
                with ui.row().classes("flex-wrap gap-4"):
                    for person in people:
                        with ui.card().classes("w-64"):
                            # Checkbox for merge selection
                            def toggle_merge(e, pid=person["id"]):
                                if e.value:
                                    merge_selection.append(pid)
                                elif pid in merge_selection:
                                    merge_selection.remove(pid)

                            ui.checkbox("Select", on_change=toggle_merge).classes("mb-2")

                            # Thumbnail
                            if person.get("thumbnail_image_path"):
                                thumb_path = Path(person["thumbnail_image_path"])
                                if thumb_path.exists():
                                    ui.image(str(thumb_path)).classes("w-full h-48 object-cover rounded")
                                else:
                                    ui.label("No thumbnail").classes("h-48 flex items-center justify-center bg-gray-200")
                            else:
                                ui.label("No thumbnail").classes("h-48 flex items-center justify-center bg-gray-200")

                            # Info
                            with ui.column().classes("p-2"):
                                name = person.get("name") or f"Person {person['person_index']}"

                                # Editable name
                                name_input = ui.input(value=name).classes("w-full")

                                async def save_name(e, pid=person["id"], inp=name_input):
                                    await api_client.rename_person(
                                        selected_album_id.value, pid, inp.value
                                    )
                                    ui.notify("Name saved!")

                                name_input.on("blur", save_name)

                                ui.label(f"{person['face_count']} faces in {person['image_count']} images").classes("text-sm text-gray-500")

                                # View images button
                                async def view_images(pid=person["id"]):
                                    ui.navigate.to(f"/person/{selected_album_id.value}/{pid}")

                                ui.button("View Images", on_click=view_images).props("flat dense")

        selected_album_id.on("change", lambda: asyncio.create_task(load_people()))

        if app_state.current_album:
            await load_people()


@ui.page("/person/{album_id}/{person_id}")
async def person_detail_page(album_id: str, person_id: str):
    """Person detail page - view all images of a person."""
    create_header()

    with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
        person = await api_client.get_person(album_id, person_id)

        name = person.get("name") or f"Person {person['person_index']}"
        ui.label(name).classes("text-2xl font-bold mb-2")
        ui.label(f"{person['face_count']} faces in {person['image_count']} images").classes("text-gray-500 mb-4")

        ui.button("Back to People", on_click=lambda: ui.navigate.to("/people")).props("flat")

        # All face instances
        ui.label("All Appearances").classes("text-lg font-semibold mt-4 mb-2")

        images = await api_client.get_person_images(album_id, person_id)

        with ui.row().classes("flex-wrap gap-4"):
            for img_data in images:
                path = Path(img_data["image_path"])
                if path.exists():
                    with ui.card().classes("w-48"):
                        ui.image(str(path)).classes("w-full h-36 object-cover rounded")
                        ui.label(path.name).classes("text-xs truncate p-1")
                        ui.label(f"{img_data['face_count']} face(s)").classes("text-xs text-gray-500 px-1")


ui.run(title="Album Organizer", port=8080, storage_secret="sim-bench-secret-key")
