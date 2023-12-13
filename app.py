from main import *

import streamlit as st
import os
import zipfile
import shutil

# 初始化会话状态
if 'processed' not in st.session_state:
    st.session_state.processed = False

def clean_dir(base_name):
    origin_zip_file = f"{base_name}.zip"
    processed_zip_file = f"{base_name}-注释后.zip"
    process_folder = f"{base_name}-注释后"
    os.remove(origin_zip_file)
    os.remove(processed_zip_file)
    shutil.rmtree(base_name)
    shutil.rmtree(process_folder)

def main():
    st.title('基于大语言模型的代码注释器')

    with st.expander("ℹ️ 说明"):
            st.markdown(
                """
                **西南大学RISE实验室**  
                *注意*：必须上传代码的.zip压缩文件。

                *说明*：很多程序员在写代码的时候经常会忘掉写代码注释，该程序就是为了解决这一个问题，使用大语言模型（LLM）给程序写上符合标准且易读的的代码注释。  
                并且保证代码结构不被破坏。支持Python,Java,Cpp,Js等 :) 
                """,
                unsafe_allow_html=False
        )
    # Upload Zip File
    uploaded_file = st.file_uploader("请上传代码库的.zip文件", type="zip")
    if uploaded_file is not None and not st.session_state.processed:
        # Save the uploaded zip file
        zipfile_name = uploaded_file.name

        with open(zipfile_name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract Zip File
        extracted_folder = zipfile_name.replace(".zip", "") # pass as base_name
        with zipfile.ZipFile(zipfile_name, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)
        
        # Define new repository path
        new_repo_path = extracted_folder + '-注释后'

        # 使用加载指示器
        with st.spinner('正在对代码进行注释，请稍候...'):
            process_repository(extracted_folder, new_repo_path)

        st.session_state.processed = True
        # Process Repository
        # process_repository("extracted_folder", new_repo_path)

        # Zip New Repository
        shutil.make_archive(new_repo_path, 'zip', new_repo_path)

        # Download Zip File
        with open(f"{new_repo_path}.zip", "rb") as f:
            st.download_button(
                label="下载注释后的代码库",
                data=f,
                file_name=f"{new_repo_path}.zip",
                mime="application/zip"
            )
        # Delete all the files
        clean_dir(extracted_folder)
        uploaded_file = None
            # 提供一个按钮来重置状态，允许用户上传新的文件
    # if st.session_state.processed:
    #     if st.button('下载上一个文件后，点击可上传另一个文件'):
    #         st.session_state.processed = False
            # 这将刷新页面并重置状态


if __name__ == "__main__":
    main()
