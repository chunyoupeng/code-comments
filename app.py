from main import *

import streamlit as st
import os
import zipfile
import shutil


def main():
    st.title('代码注释器')

    # Upload Zip File
    uploaded_file = st.file_uploader("请上传代码库的.zip文件", type="zip")
    if uploaded_file is not None:
        # Save the uploaded zip file
        with open("uploaded_zip.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract Zip File
        with zipfile.ZipFile("uploaded_zip.zip", 'r') as zip_ref:
            zip_ref.extractall("extracted_folder")

        # Define new repository path
        new_repo_path = "new_repository"

        # 使用加载指示器
        with st.spinner('正在对代码进行注释，请稍候...'):
            process_repository("extracted_folder", new_repo_path)

        # Process Repository
        # process_repository("extracted_folder", new_repo_path)

        # Zip New Repository
        shutil.make_archive("processed_repository", 'zip', new_repo_path)

        # Download Zip File
        with open("processed_repository.zip", "rb") as f:
            st.download_button(
                label="Download Processed Repository",
                data=f,
                file_name="processed_repository.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()
