import streamlit as st


def render_about_content():
    st.html(
        """
        <div class = "box">
            <div style="background-color: black; text-align: justify; border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; color: white;">
                <div class="box-header">What is Geoeffectiveness?</div>
                <p>Geoeffectiveness refers to the impact of solar and geomagnetic activity on Earth's environment and technological systems. Understanding and predicting geoeffective events, such as geomagnetic storms, is crucial for mitigating their effects on satellites, power grids, and communication systems.</p>
                <p>Frontier Development Lab (FDL) contributes to this field using advanced Machine Learning (ML) and Artificial Intelligence (AI) techniques to enhance our understanding and prediction capabilities, ensuring better preparedness for geoeffective events. Through ML models like the Solar wind High-speed Enhancements And Transients Handler (SHEATH) and the Deep leArninG Geomagnetic pErtuRbation (DAGGER), FDL is able to identify, analyze, and forecast solar wind transients and geomagnetic perturbations with unprecedented accuracy, providing critical insights that help mitigate the risks associated with space weather events.</p>

                <img src="https://storage.googleapis.com/india-jackson-1/dagger_sheath.png" style="width:100%; height:auto; margin-bottom: 20px;">

                <p>The Solar wind High-speed Enhancements And Transients Handler (SHEATH) project, initiated during FDL 2020, focuses on identifying and analyzing high-speed solar wind streams and their interactions with Earth's magnetosphere. This project aims to enhance our understanding of solar wind dynamics and their impact on geomagnetic activity. Following SHEATH, the Deep leArninG Geomagnetic pErtuRbation (DAGGER) project was developed during FDL 2023. It utilizes deep learning techniques to predict geomagnetic disturbances caused by solar wind and other space weather phenomena. The objective of DAGGER is to improve space weather forecasting capabilities and provide timely alerts for potential geoeffective events. Building on the progress of SHEATH and DAGGER, the Continuous Learning (CL) initiative integrates real-time data acquisition with machine learning to continually refine model accuracy and predictive power. Effective implementation of CL depends on the availability of Near Real-Time (NRT) data, which is crucial for updating models with the latest observations and enhancing forecast reliability..</p>

            </div>

            <div style="text-align: justify; border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; color: white;">

            <div class="box-header">Near Real Time (NRT) Data</div>
                <p>Near Real-Time (NRT) data refers to data that is collected and processed with minimal delay, allowing for real-time analysis and decision-making. NRT data is crucial for the success of Continuous Learning, as it provides the most up-to-date information available. Near Real-Time Solar Dynamic Observatory Machine Learning (NRT SDOML) involves using real-time data from the Solar Dynamic Observatory (SDO) to train and update machine learning models. This approach helps in monitoring and predicting solar activities that can impact space weather. L1 data refers to data collected from the first Lagrangian point (L1), where the gravitational forces of the Earth and the Sun balance the orbital motion of a satellite. This makes L1 an ideal location for monitoring solar wind and other space weather phenomena.</p>

                <!---------------------------------------------- SDO ----------------------------------------->
                <div style="background-color: black; border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; display: flex; justify-content: space-between; align-items: center; color: white;">

                    <div style="flex: 1; padding-right: 20px; text-align: justify;">
                        <div class="box-header">SDO</div>
                        <p>The Solar Dynamics Observatory (SDO) spacecraft, operated by NASA, continuously observes the Sun and sends its Level 0 (L0) data to ground stations like the White Sands Complex in New Mexico. From there, the data is transmitted to NASA’s Goddard Space Flight Center (GSFC), where it is ingested into the SDO Data Processing Center. Once the L0 data is received at GSFC, it undergoes processing to generate higher-level data products, such as Level 1 and Level 2 data. These products include detailed measurements of the Sun’s magnetic field, solar irradiance, and images of the Sun in various ultraviolet wavelengths. The processed data is then made available to the scientific community and the public through the Joint Science Operations Center (JSOC) and other NASA data repositories, contributing to a deeper understanding of solar dynamics and space weather forecasting.</p>
                        <p>Credit: <a href = "https://www.eoportal.org/satellite-missions/sdo#sdo-solar-dynamics-observatory" target="_blank">NASA</a></p>
                    </div>

                    <div style="flex: 1; text-align: center;">
                        <img src="https://storage.googleapis.com/india-jackson-1/SDO_Auto30_highlighted.png" style="width: 100%; max-width: 400px; height: auto;">
                    </div>

                </div>

                <!---------------------------------------------- ACE ----------------------------------------->
                <div style="background-color: black; border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; display: flex; color: white; justify-content: space-between; align-items: center;">

                    <div style="flex: 1; color: white; padding-right: 20px; text-align: justify;">
                        <div class="box-header">ACE</div>
                        <p>The Advanced Composition Explorer (ACE) is located at L1, collecting solar wind data, including speed, density, and magnetic field strength. ACE spacecraft sends its Level 0 (L0) data to the Deep Space Network (DSN), which is managed by NASA's Jet Propulsion Laboratory (JPL). The DSN is a network of large antennas and communication facilities located around the world, which are used to communicate with spacecraft that are beyond low Earth orbit. Once the L0 data is received by the DSN, it is sent to the Space Physics Data Facility (SPDF) at NASA's Goddard Space Flight Center. The data is then processed into higher-level data products (e.g., Level 1 and Level 2 data) and made available to the scientific community through the SPDF and other NASA data centers. The data is downlinked in near real-time to Earth, where it is processed and ingested into various space weather models.</p>
                        <p>Credit: <a href = "https://www.eoportal.org/satellite-missions/ace" target="_blank">NASA</a></p>
                    </div>

                    <div style="flex: 1; text-align: center;">
                        <img src="https://storage.googleapis.com/india-jackson-1/ACE_Auto1B_highlighted_1.png" style="width: 100%; max-width: 400px; height: auto;">
                    </div>

                </div>

                <!---------------------------------------------- DSCOVR ----------------------------------------->
                <div style="background-color: black; border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; display: flex; justify-content: space-between; align-items: center;">

                    <div style="flex: 1; padding-right: 20px; text-align: justify;">
                        <div class="box-header">DSCOVR</div>
                        <p>The Deep Space Climate Observatory (DSCOVR) is also located at L1, providing critical data on solar wind and magnetic fields. Like ACE, DSCOVR’s data is downlinked in near real-time and used to monitor and predict space weather conditions. DSCOVR (Deep Space Climate Observatory) spacecraft sends its Level 0 (L0) data to NOAA’s Satellite Operations Facility (NSOF) after it is received by ground stations, such as those at the NOAA Wallops Command and Data Acquisition Station (WCDAS) in Virginia. The data is then transmitted to NOAA's Space Weather Prediction Center (SWPC) for processing. Once the L0 data is received at NSOF, it is processed into higher-level data products, including Level 1 and Level 2 data, which provide detailed information on solar wind parameters and magnetic field measurements. These processed data products are made available to the scientific community and the public through NOAA's data repositories and other relevant platforms.</p>
                        <p>Credit: <a href = "https://www.eoportal.org/satellite-missions/dscovr" target="_blank">NASA</a></p>
                    </div>

                    <div style="flex: 1; text-align: center;">
                        <img src="https://storage.googleapis.com/india-jackson-1/DSCOVR_Auto27_highlighted.png" style="width: 100%; max-width: 400px; height: auto;">
                    </div>

                </div>

                <!---------------------------------------------- GEOMAGNETIC INDICES ----------------------------------------->

                <div style="background-color: black; text-align: justify; border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px;">
                    <div class="box-header">Geomagnetic Indices (Hp30, ap30, Kp, Fadj)</div>
                    <p>These indices are collected by various instruments located on Earth. They measure geomagnetic activity, providing essential information on how solar wind interacts with Earth's magnetic field. This data is crucial for understanding the impact of solar events on Earth's environment. Geomagnetic indices like Kp, ap, and Hp30 are vital tools for monitoring and predicting space weather events that can impact Earth's technological systems. Additionally, the F10.7 solar radio flux adjusted for the Sun-Earth distance (Fadj) serves as a crucial indicator of solar activity, providing insights into the solar emissions that influence Earth's upper atmosphere and ionosphere.</p>

                    <p>The Kp index, for instance, is calculated using measurements from 13 key geomagnetic observatories around the world. These observatories, such as Niemegk in Germany, Førde in Norway, Hartland in the UK, Wingst in Germany, and Eskdalemuir in the UK, among others, provide a consistent and global measure of geomagnetic activity.</p>

                    <p>The ap index, which is a linear equivalent of the Kp index, also relies on data from these observatories to offer a more detailed view of geomagnetic disturbances in nanoteslas (nT). High-latitude geomagnetic activity, which is crucial for understanding auroral activity and polar disturbances, is captured by the Hp30 index. This index is derived from magnetometer data collected at observatories in high-latitude regions such as Qaanaaq in Greenland, Tromsø in Norway, Barrow in the USA, Vostok in Antarctica, and McMurdo in Antarctica. These stations are ideally situated to monitor the intense geomagnetic activity often seen in these regions.</p>

                    <p>In addition to these indices, other specialized indices like Dst (Disturbance Storm Time) and AE (Auroral Electrojet) are generated from different sets of ground-based magnetometers, focusing on specific latitudes or regions to provide insights into storm-time disturbances and auroral zone activities. The Fadj index, collected at the Dominion Radio Astrophysical Observatory in Penticton, Canada, complements these geomagnetic measurements by providing a critical measure of solar radio flux, helping to monitor solar activity and its effects on Earth's atmosphere. The data from these observatories are collected, processed, and distributed by national and international data centers, such as NOAA’s Space Weather Prediction Center (SWPC) and the World Data Centers for Geomagnetism, playing a critical role in space weather forecasting and research.</p>
                    <p>Credit: <a href = "https://www.gfz-potsdam.de/en/section/geomagnetism/infrastructure/geomagnetic-observatories" target="_blank">Helmholtz Centre: Potsdam</a></p>
                    <img src="https://storage.googleapis.com/india-jackson-1/geomagnetic_observatories_1.png" style="width: 100%; height: auto;" alt="Niemegk Observatory">

                </div>
            </div>

            <div style="background-color: black; border: 2px solid #ffffff; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); transition: 0.3s; width: 100%; margin: auto; margin-top: 20px; padding: 20px; color: black;">
                <!--<div class="box-header">FDL-Geo Teams</div>-->
                    <div style="display: grid; grid-template-columns: 1fr; gap: 20px;">

                        <div class="box-header">FDL-2024 GeoCloak</div>
                        <div><img src="https://storage.googleapis.com/india-jackson-1/geocloak_2.png" style="width:100%; height:auto;"></div>
                        <p></p>

                        <div class="box-header">FDL-2023 SHEATH</div>
                        <div><img src="https://storage.googleapis.com/india-jackson-1/SHEATH_3.png" style="width:100%; height:auto;"></div>
                        <p></p>

                        <div class="box-header">FDL-2020 DAGGER</div>
                        <div><img src="https://storage.googleapis.com/india-jackson-1/DAGGER.png" style="width:100%; height:auto;"></div>
                        <p></p>

                        <!--
                        <div class="box-header">FDL-2025 <new_team_name></div>
                        <div><img src="path/to/team_image_4.png" style="width:100%; height:auto;"></div>
                        <p></p>
                        -->

                    </div>
                </div>

        </div>
        """,
    )
