"""
==============================================================================
VGREEN AGRI-INTELLIGENCE PLATFORM (VAN EXECUTIVE PROTOTYPE)
==============================================================================
Author: Muhammad Arsalan | Project: Proprietary Agri Data Pipeline
Target Area: Kasur / Phool Nagar Smallholder Cluster (1-4 Acre Parcels)
"""

import os
import subprocess
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="VGreen Agri-Intelligence Portal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom VGreen Enterprise CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #065F46;
        padding-bottom: 5px;
        border-bottom: 2px solid #10B981;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stButton>button {
        background-color: #059669;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. PATH DEFINITIONS & DATA LOADING
# ==============================================================================
BASE_DIR = "/home/muhammadarsalan/Desktop/FYP"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
CSV_INDEX_PATH = os.path.join(DATASET_DIR, "vgreen_dataset_index.csv")
FIELD_CARDS_DIR = os.path.join(DATASET_DIR, "field_polygon_images")
GRAPH_DIR = os.path.join(DATASET_DIR, "generated_graphs")
RASTER_DIR = os.path.join(DATASET_DIR, "super_res_rasters")
COMPOSITE_MAP_PATH = os.path.join(RASTER_DIR, "kasur_master_cluster_composite.png")

@st.cache_data
def load_dataset_index(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_data = load_dataset_index(CSV_INDEX_PATH)

# ==============================================================================
# 3. SIDEBAR BRANDING & CONTROLS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/sprout.png", width=70)
st.sidebar.title("VGreen Intelligence")
st.sidebar.caption("Vital Agri Nutrients | Kasur Cluster")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Pipeline Operations")

if st.sidebar.button("🔄 Execute Live Data Pipeline"):
    with st.spinner("Fetching live satellite tiles, applying 4x AI Super-Resolution, and refreshing analytics..."):
        try:
            res = subprocess.run(["python", os.path.join(BASE_DIR, "live_pipeline.py")], capture_output=True, text=True)
            st.cache_data.clear()
            st.sidebar.success("Pipeline Executed Successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Execution Error: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Dataset Filters")

if df_data is not None:
    crop_options = ["All"] + list(df_data["Predicted_Crop"].unique())
    selected_crop = st.sidebar.selectbox("Filter by Crop Type", crop_options)
    
    status_options = ["All"] + list(df_data["Status"].unique())
    selected_status = st.sidebar.selectbox("Filter by Field Status", status_options)
    
    # Filter logic
    filtered_df = df_data.copy()
    if selected_crop != "All":
        filtered_df = filtered_df[filtered_df["Predicted_Crop"] == selected_crop]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["Status"] == selected_status]
else:
    filtered_df = pd.DataFrame()

st.sidebar.markdown("---")
st.sidebar.info("💡 **Enterprise Asset:** VGreen Ground-Truth Spatial Boundary & AI Forecasting Engine.")

# ==============================================================================
# 4. MAIN DASHBOARD CONTENT
# ==============================================================================
st.markdown("<div class='main-header'>🌱 Vital Agri Nutrients (VGreen) — Smallholder Cluster Platform</div>", unsafe_allow_html=True)

if df_data is None:
    st.warning("⚠️ Dataset Index (`vgreen_dataset_index.csv`) not found. Please click **'Execute Live Data Pipeline'** in the sidebar to build the dataset.")
else:
    # --------------------------------------------------------------------------
    # TOP METRIC CARDS (KPIS)
    # --------------------------------------------------------------------------
    active_df = df_data[df_data["Status"] == "Active Field"]
    discarded_count = len(df_data) - len(active_df)
    total_acreage = active_df["Acreage_Acres"].sum()
    
    total_nitro = active_df["VGreen_NitroPlus_Bags"].sum()
    total_biophos = active_df["VGreen_BioPhos_Bags"].sum()
    total_sop = active_df["VGreen_SOP_Bags"].sum()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Polygons", len(df_data), help="Total digitized boundaries in Kasur cluster")
    kpi2.metric("Active Agri Fields", len(active_df), delta=f"-{discarded_count} Built-up/Roads", delta_color="inverse")
    kpi3.metric("Total Cluster Area", f"{total_acreage:.1f} Acres")
    kpi4.metric("NitroPlus Forecast", f"{total_nitro:,} Bags", help="VGreen NitroPlus required")
    kpi5.metric("BioPhos Forecast", f"{total_biophos:,} Bags", help="VGreen BioPhos required")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # TABS SECTION
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Master Cluster Map", 
        "🔬 Field Inspector & Phenology", 
        "📊 Supply Chain & Fertilizer Forecast", 
        "📄 Dataset Master Index"
    ])

    # ==========================================================================
    # TAB 1: MASTER CLUSTER OVERLAY
    # ==========================================================================
    with tab1:
        st.subheader("Kasur/Phool Nagar Cluster — 2.5m AI Super-Resolved Overlay")
        st.caption("AI-sharpened satellite raster integrated with QGIS digitized ground-truth vector boundaries.")
        
        col_map, col_summary = st.columns([2.5, 1])
        
        with col_map:
            if os.path.exists(COMPOSITE_MAP_PATH):
                comp_img = Image.open(COMPOSITE_MAP_PATH)
                st.image(comp_img, caption="Master Composite Map (Green=Active Fields, Red=Built-up Discarded)", use_container_width=True)
            else:
                st.error("Master Composite Map not generated yet. Run the live pipeline.")

        with col_summary:
            st.markdown("#### 🌾 Crop Distribution")
            crop_counts = active_df["Predicted_Crop"].value_counts().reset_index()
            crop_counts.columns = ["Crop", "Count"]
            
            fig_pie = px.pie(
                crop_counts, 
                values="Count", 
                names="Crop", 
                color="Crop",
                color_discrete_map={
                    "Wheat": "#10B981", 
                    "Sugarcane": "#F59E0B", 
                    "Rice": "#3B82F6", 
                    "Maize": "#8B5CF6"
                },
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=220)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("#### 🏗️ Boundary Integrity Check")
            st.write(f"• **Valid Agricultural Plots:** `{len(active_df)}` fields")
            st.write(f"• **Discarded Non-Agri/Houses:** `{discarded_count}` plots")
            st.info("Peak 12-month NDVI thresholding (<0.25) automatically filters out non-agricultural encroached areas.")

    # ==========================================================================
    # TAB 2: FIELD INSPECTOR & PHENOLOGY
    # ==========================================================================
    with tab2:
        st.subheader("Individual Field Polygon Analytics")
        
        field_list = filtered_df["Field_ID"].tolist() if not filtered_df.empty else df_data["Field_ID"].tolist()
        selected_field_id = st.selectbox("Select Field ID for Detailed Inspection", field_list)
        
        field_row = df_data[df_data["Field_ID"] == selected_field_id].iloc[0]
        
        col_card, col_graph, col_meta = st.columns([1.2, 1.8, 1])
        
        with col_card:
            st.markdown(f"##### 📸 2.5m AI Super-Res Card ({selected_field_id})")
            card_file = os.path.join(FIELD_CARDS_DIR, str(field_row["Card_Image"]))
            if os.path.exists(card_file):
                st.image(Image.open(card_file), use_column_width=True)
            else:
                st.warning("Card image missing.")

        with col_graph:
            st.markdown(f"##### 📈 12-Month NDVI Phenology Profile")
            graph_file = os.path.join(GRAPH_DIR, str(field_row["NDVI_Graph"]))
            if os.path.exists(graph_file):
                st.image(Image.open(graph_file), use_column_width=True)
            else:
                st.warning("Phenology graph missing.")

        with col_meta:
            st.markdown("##### 📋 Field Specifications")
            st.write(f"**Field ID:** `{field_row['Field_ID']}`")
            st.write(f"**Status:** `{field_row['Status']}`")
            st.write(f"**Area:** `{field_row['Acreage_Acres']} Acres`")
            st.write(f"**Predicted Crop:** `{field_row['Predicted_Crop']}`")
            st.write(f"**Peak NDVI:** `{field_row['Max_NDVI']:.2f}`")
            st.write(f"**Coordinates:** `{field_row['Latitude']:.4f}° N, {field_row['Longitude']:.4f}° E`")
            
            st.markdown("---")
            st.markdown("##### 🧪 VGreen Prescription")
            if field_row['Status'] == 'Active Field':
                st.success(f"• **NitroPlus:** {field_row['VGreen_NitroPlus_Bags']} Bags\n"
                           f"• **BioPhos:** {field_row['VGreen_BioPhos_Bags']} Bags\n"
                           f"• **SoluPotash (SOP):** {field_row['VGreen_SOP_Bags']} Bags")
            else:
                st.error("No fertilizer allocated (Non-Agri/Built-up Area).")

    # ==========================================================================
    # TAB 3: SUPPLY CHAIN & FERTILIZER FORECAST
    # ==========================================================================
    with tab3:
        st.subheader("Regional Supply Chain & Fertilizer Logistics Forecasting")
        st.caption("Commercial volume estimates aggregated by crop type across the Kasur smallholder cluster.")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### 📦 Fertilizer Demand by Crop Type (Bags)")
            crop_fert = active_df.groupby("Predicted_Crop")[
                ["VGreen_NitroPlus_Bags", "VGreen_BioPhos_Bags", "VGreen_SOP_Bags"]
            ].sum().reset_index()
            
            fig_bar = px.bar(
                crop_fert, 
                x="Predicted_Crop", 
                y=["VGreen_NitroPlus_Bags", "VGreen_BioPhos_Bags", "VGreen_SOP_Bags"],
                barmode="group",
                labels={"value": "Total Bags Needed", "Predicted_Crop": "Crop Type", "variable": "VGreen Product"},
                color_discrete_sequence=["#10B981", "#3B82F6", "#F59E0B"]
            )
            fig_bar.update_layout(height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            st.markdown("##### 📐 Cultivated Acreage Breakdown")
            fig_acre = px.bar(
                active_df.groupby("Predicted_Crop")["Acreage_Acres"].sum().reset_index(),
                x="Predicted_Crop",
                y="Acreage_Acres",
                color="Predicted_Crop",
                color_discrete_map={"Wheat": "#10B981", "Sugarcane": "#F59E0B", "Rice": "#3B82F6", "Maize": "#8B5CF6"},
                labels={"Acreage_Acres": "Total Cultivated Area (Acres)"}
            )
            fig_acre.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_acre, use_container_width=True)

    # ==========================================================================
    # TAB 4: MASTER DATASET INDEX TABLE
    # ==========================================================================
    with tab4:
        st.subheader("VGreen Master Proprietary Dataset Index")
        st.write(f"Displaying **{len(filtered_df)}** records matching active filters.")
        
        st.dataframe(
            filtered_df,
            column_config={
                "Card_Image": None,
                "NDVI_Graph": None,
                "Max_NDVI": st.column_config.NumberColumn("Peak NDVI", format="%.2f"),
                "Acreage_Acres": st.column_config.NumberColumn("Area (Acres)", format="%.2f"),
            },
            use_container_width=True,
            height=400
        )
        
        # CSV Export Download Button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Selected VGreen Dataset (CSV)",
            data=csv_data,
            file_name="vgreen_filtered_dataset.csv",
            mime="text/csv"
        )
