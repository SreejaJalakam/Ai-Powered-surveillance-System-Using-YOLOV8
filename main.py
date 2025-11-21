#!/usr/bin/env python3
"""
AbnoGuard - AI Surveillance for Abnormality & Abandoned Objects
Main entry point using YOLOv8 for enhanced detection
"""

import streamlit as st
import os
import sys
from video_runner import VideoRunner
from utils import setup_directories, check_dependencies

def main():
    """Main entry point for AbnoGuard"""
    st.title("🚀 AbnoGuard - AI Surveillance System")
    st.markdown("🔍 Powered by YOLOv8 for Enhanced Detection")
    st.markdown("---")
    
    # Setup directories
    setup_directories()
    
    # Check dependencies
    if not check_dependencies():
        st.error("❌ Dependencies check failed. Please install required packages.")
        return
    
    # Streamlit file uploader (replaces Tkinter dialog)
    st.subheader("📁 Select a video file for analysis")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "wmv"]
    )
    
    if uploaded_file is None:
        st.info("Please upload a video file to start analysis.")
        return
    
    # Save uploaded file temporarily
    temp_video_path = os.path.join("temp_video", uploaded_file.name)
    os.makedirs("temp_video", exist_ok=True)
    
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success(f"✅ Selected video: {uploaded_file.name}")
    st.write(f"📊 File size: {len(uploaded_file.getbuffer()) / (1024*1024):.1f} MB")
    
    # Run video analysis
    if st.button("🎬 Start Video Analysis"):
        try:
            st.write("💡 YOLOv8 provides better accuracy and more object classes")
            st.info("Processing video... Please wait ⏳")
            
            runner = VideoRunner(temp_video_path)
            runner.run()
            
            st.success("🎯 AbnoGuard analysis complete!")
            st.info("📁 Check the 'outputs/' folder for results and alerts")
        
        except Exception as e:
            st.error(f"❌ Error during video analysis: {e}")

if __name__ == "__main__":
    main()
