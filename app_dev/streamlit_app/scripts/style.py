# style.py
import streamlit as st


def render_custom_style():
    st.html(
        """
        <style>
        header[data-testid="stHeader"] {
            background-image: url("https://storage.googleapis.com/india-jackson-1/Geocloak-04.png");
            /*background-image: url("https://storage.googleapis.com/india-jackson-1/NEW_BANNER_3.png");*/
            background-size: cover;
            background-position: center;
            width: 100%;
            height: 100px;  /*Adjust this as needed */
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        section[data-testid="stSidebar"] {
            top: 100px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
         /*   width: 100%;*/
        }
        
        div[data-testid="stVerticalBlock"] {
            width: 100vh;
            width: 500px;
        }

        section.main.st-emotion-cache-bm2z3a.ea3mdgi8 {
            padding-right: 700px;  /* Adjust the padding as needed */
        }
        /*
        div.st-emotion-cache-0.e1f1d6gn0 {
            width: 100vh;
            text-align: center;
            padding-top: 100px;
            border: 2px solid grey;
            border-radius: 10px;
            /*padding: 20px;*/
            margin-bottom: 20px;
            background-color: #1e1e1e;
            margin: 20px auto;
        }
        */
        div.st-emotion-cache-1wmy9hl.e1f1d6gn1 {
            width: 100vh;
            text-align: center;
            padding-top: 100px;
            border: 2px solid grey;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #1e1e1e;
            margin: 20px auto;
            color: white;
        }
           
        
        div.st-emotion-cache-ocqkz7.e1f1d6gn5 {
            width: 100vh;  /* Adjust the padding as needed */
            text-align: center;
            color: white;
        }
        
        div.st-emotion-cache-ef2wwy.e1f1d6gn2 {
            padding-top: 100px;
            border: 2px solid grey;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #1e1e1e;
            margin: 20px auto;
            color: white;
        }
        
        div.st-emotion-cache-keje6w.e1f1d6gn3 {
            padding-top: 100px;
            border: 2px solid grey;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            /*background-color: #1e1e1e;*/
            background-color: #000000;
            margin: 20px auto;
            color: white;
        }

        .box {
            background: #1e1e1e;
            border: 2px solid #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            transition: 0.3s;
            width: 100vh;
            height: 100%;
            margin: 20px auto; /* Center the box horizontally and add top margin */
            align-items: center;
            color: white;
            padding: 20px;  /*Add padding inside the box */
        }
        .box:hover {
            box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
        }
        .box-header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
            color: white;
        }
        </style>
        """,
    )
