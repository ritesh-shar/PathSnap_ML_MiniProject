
import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="PathSnap | Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
)

st.title("PathSnap")
st.subheader("Distributed Map-Matching & Route Reconstruction")
st.subheader("Group 6")
st.caption("ML-powered GPS trajectory reconstruction using MLP and Viterbi decoding")

# Laptop 1's address goes here when running on Laptop 2.
default_url = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")

with st.sidebar:
    st.header("API Configuration")
    api_url = st.text_input("Backend API URL", value=default_url).rstrip("/")
    st.caption("Enter Laptop 1's IPv4 address when using another laptop.")

    if st.button("Check API connection"):
        try:
            response = requests.get(f"{api_url}/", timeout=5)
            response.raise_for_status()
            st.success("Backend connected!")
        except requests.RequestException as exc:
            st.error(f"Connection failed: {exc}")


@st.cache_data(ttl=15)
def get_trips(url):
    response = requests.get(f"{url}/trips", timeout=10)
    response.raise_for_status()
    return response.json()["trips"]


try:
    trips = get_trips(api_url)
except requests.RequestException as exc:
    st.error(f"Could not retrieve trips from the API: {exc}")
    st.stop()

if not trips:
    st.warning("No trips were found in the backend evaluation directory.")
    st.stop()

st.markdown("### Select a GPS trip")

selected_trip = st.selectbox("Available trips", trips)

if st.button("Reconstruct route", type="primary"):
    with st.spinner("Generating candidates and reconstructing route..."):
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"trip_id": selected_trip},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            st.session_state["prediction"] = result

        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
result = st.session_state.get("prediction")

if result and result["trip_id"] == selected_trip:

    st.divider()
    st.markdown(f"### Results for trip `{selected_trip}`")

    metrics = result["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Arc accuracy",
        f"{metrics['arc_accuracy']:.2%}"
        if metrics["arc_accuracy"] is not None else "N/A"
    )
    c2.metric("GPS observations", metrics["observations"])
    c3.metric("Candidate arcs", metrics["candidate_count"])
    c4.metric(
        "Prediction time",
        f"{metrics['prediction_time_seconds']:.3f}s"
    )

    c5, c6 = st.columns(2)
    c5.metric(
        "Candidate miss rate",
        f"{metrics['candidate_miss_rate']:.2%}"
        if metrics["candidate_miss_rate"] is not None else "N/A"
    )
    c6.metric("Evaluated observations", metrics["evaluated_observations"])

    gps_coordinates = result["gps_coordinates"]
    predicted_coordinates = result["predicted_coordinates"]
    road_network = result["road_network"]

    st.markdown("### Route reconstruction map")

    if len(predicted_coordinates) >= 2 and len(gps_coordinates) >= 2:

        gps_points = [[lat, lon] for lon, lat in gps_coordinates]
        predicted_points = [
            [lat, lon] for lon, lat in predicted_coordinates
        ]

        center = gps_points[len(gps_points) // 2]

        route_map = folium.Map(
            location=center,
            zoom_start=14,
            control_scale=True
        )

        road_layer = folium.FeatureGroup(
            name="Road network",
            show=True
        )

        for segment in road_network:
            points = [[lat, lon] for lon, lat in segment]
            folium.PolyLine(
                points,
                color="gray",
                weight=1,
                opacity=0.20
            ).add_to(road_layer)

        road_layer.add_to(route_map)

        gps_layer = folium.FeatureGroup(name="Raw GPS trace")
        folium.PolyLine(
            gps_points,
            color="red",
            weight=3,
            opacity=0.8,
            tooltip="Raw GPS trajectory"
        ).add_to(gps_layer)
        gps_layer.add_to(route_map)

        predicted_layer = folium.FeatureGroup(name="Predicted route")
        folium.PolyLine(
            predicted_points,
            color="blue",
            weight=5,
            opacity=0.9,
            tooltip="MLP + Viterbi predicted route"
        ).add_to(predicted_layer)

        folium.Marker(
            predicted_points[0],
            tooltip="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(predicted_layer)

        folium.Marker(
            predicted_points[-1],
            tooltip="End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(predicted_layer)

        predicted_layer.add_to(route_map)

        folium.LayerControl(collapsed=False).add_to(route_map)
        route_map.fit_bounds(gps_points)

        components.html(
            route_map._repr_html_(),
            height=650,
            scrolling=False
        )

    else:
        st.warning("Not enough coordinates to display the route.")

else:
    st.info("Select a trip and click **Reconstruct route** to generate a prediction.")
