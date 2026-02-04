import numpy as np
import pandas as pd
import sklearn
import joblib
from tensorflow import keras
import streamlit as st
#from streamlit.logger import get_logger


def main_predict(txtfile, model, threshold):
    txtfile = txtfile[[txtfile.columns[1]]]
    txtfile = np.asarray(txtfile).astype(np.float32).T
    txtfile = txtfile.reshape(-1, 23)
    code2rel = {0: 'Tidak berhasil', 1: 'Berhasil'}
    
    proba_val = model.predict(txtfile)
    
    proba = proba_val*100
    np.set_printoptions(precision=2)
    proba = str(proba)[2:-2]+"%"
    predict = 1 if proba_val > threshold else 0
    #print(f"{code2rel[predict]}, dengan akurasi {str(proba)[2:-2]}")
    output_txt = code2rel[predict]
    return output_txt, proba

model = keras.models.load_model('model/')
threshold = joblib.load('model/acc_threshold.pkl')




def run():
    st.set_page_config(
        page_title="Live-birth Occurrence Prediction",
        page_icon="👶",
    )

    st.title("A Simple Web App for Predicting Live-birth Occurence")
    st.write("""Predict the live-birth occurrence before in-vitro fertilization (IVF) treatment using machine learning. 
    Get another opinion based on the medical records, or simply save cost and procedures before going into any IVF procedures 
    
    """)

    expander1 = st.expander("About this app")
    expander1.write("""
    This app was created for educational purposes only. Prediction result is generated based on a deep learning model trained using the dataset taken from Human Fertilisation & Embryology Authority. The model itself could be changed in the future for a better prediction result. 
    Hence, it is strongly advised not to take the prediction result as the main reason for deciding any actions regarding the IVF treatment.

    We only consider the patients that will receive the IVF treatment to receive some stimulation. Features needed to train the model are selected based on the recommendation from a physician's review of the following paper: "A systematic review of the quality of clinical prediction models in in-vitro fertilization".
    The input required to get the prediction result consists of those selected features.

    For more details/inquiries, contact: @harry-bpm at github
    """

    )
    expander2 = st.expander("Quick guide")
    expander2.write("""
      Please fill in the following forms based on the medical record.
      - Slide the slider button to input the data. 
      - To input a number in the form, type the number followed by the "enter" button.
      - Tick the checkbox to indicate a positive response. 
      
      Press the "predict" button to show the prediction result.
      
      """)

    with st.form(key='columns_in_form'):
        col1, col2, col3, col4 = st.columns(4)
        col1.subheader("General Information")
        with col1:
            age_at_treatment_status = st.select_slider('Select age range', options=['18-34', '35-37', '38-39', '40-42', '43-44', '45-50'])
            no_prev_treatment_status = st.select_slider( "Total Number of Previous treatments, Both IVF and DI at clinic", options=['0', '1', '2', '3', '4', '5','>5'])
            no_ivf_pregnancy_status = st.select_slider("Total number of IVF pregnancies", options=['0', '1', '2', '3', '4', '5'])
            no_prev_ivf_cycle_status = st.select_slider("Total Number of Previous IVF cycles", options=['0', '1', '2', '3', '4', '5','>5'])
            embryos_transfer_status  = st.select_slider("Embryos Transfered" , options=['0', '1', '2', '3'])
            total_embryo = st.number_input( "Total Embryos Created")

            if age_at_treatment_status=="18-34":
                age_at_treatment=0
            elif age_at_treatment_status=="35-37":
                age_at_treatment=1
            elif age_at_treatment_status=="38-39":
                age_at_treatment=2
            elif age_at_treatment_status=="40-42":
                age_at_treatment=3
            elif age_at_treatment_status=="43-44":
                age_at_treatment=4
            else:
                age_at_treatment=5

            if no_prev_treatment_status=="0":
                no_prev_treatment=0
            elif no_prev_treatment_status=="1":
                no_prev_treatment=1
            elif no_prev_treatment_status=="2":
                no_prev_treatment=2
            elif no_prev_treatment_status=="3":
                no_prev_treatment=3
            elif no_prev_treatment_status=="4":
                no_prev_treatment=4
            elif no_prev_treatment_status=="5":
                no_prev_treatment=5
            else:
                age_at_treatment=6
            
            if no_prev_ivf_cycle_status=="0":
                no_prev_ivf_cycle=0
            elif no_prev_ivf_cycle_status=="1":
                no_prev_ivf_cycle=1
            elif no_prev_ivf_cycle_status=="2":
                no_prev_ivf_cycle=2
            elif no_prev_ivf_cycle_status=="3":
                no_prev_ivf_cycle=3
            elif no_prev_ivf_cycle_status=="4":
                no_prev_ivf_cycle=4
            elif no_prev_ivf_cycle_status=="5":
                no_prev_ivf_cycle=5
            else:
                no_prev_ivf_cycle=6

            if no_ivf_pregnancy_status=="0":
                no_ivf_pregnancy=0
            elif no_ivf_pregnancy_status=="1":
                no_ivf_pregnancy=1
            elif no_ivf_pregnancy_status=="2":
                no_ivf_pregnancy=2
            elif no_ivf_pregnancy_status=="3":
                no_ivf_pregnancy=3
            elif no_ivf_pregnancy_status=="4":
                no_ivf_pregnancy=4
            else:
                no_ivf_pregnancy=5

            if embryos_transfer_status=="0":
                embryos_transfer=0
            elif embryos_transfer_status=="1":
                embryos_transfer=1
            elif embryos_transfer_status=="2":
                embryos_transfer=2
            else:
                embryos_transfer=3

        

        col2.subheader("Type of Infertility")
        with col2:
            female_primary_status = st.checkbox( "Female Primary" )
            female_secondary_status = st.checkbox( "Female Secondary ")
            male_primary_status = st.checkbox( "Male Primary" )
            male_secondary_status = st.checkbox( "Male Secondary" )
            couple_primary_status = st.checkbox( "Couple Primary" )
            couple_secondary_status = st.checkbox( "Couple Secondary" )

            couple_secondary=1 if couple_secondary_status else 0
            female_primary=1 if female_primary_status else 0
            female_secondary=1 if female_secondary_status else 0
            male_primary=1 if male_primary_status else 0
            male_secondary=1 if male_secondary_status else 0
            couple_primary=1 if couple_primary_status else 0
    
        col3.subheader("Cause of Infertility I")
        with col3:    
            tubal_disease_status = st.checkbox( "Tubal disease" )
            ovulatory_disorder_status = st.checkbox( "Ovulatory Disorder" )
            male_factor_status = st.checkbox( "Male Factor" )
            patient_unexplained_status = st.checkbox( "Patient Unexplained" )
            endometriosis_status = st.checkbox( "Endometriosis ")

            tubal_disease=1 if tubal_disease_status else 0
            ovulatory_disorder=1 if ovulatory_disorder_status else 0
            male_factor=1 if male_factor_status else 0
            patient_unexplained=1 if patient_unexplained_status else 0
            endometriosis=1 if endometriosis_status else 0


        col4.subheader("Cause of Infertility II")
        with col4:  
            cervical_factors_status = st.checkbox( "Cervical factors ")
            female_factors_status = st.checkbox( "Female Factors" )
            partner_sperm_concentration_status = st.checkbox( "Partner Sperm Concentration" )
            partner_sperm_morphology_status = st.checkbox( "Partner Sperm Morphology" )
            partner_sperm_motility_status = st.checkbox( " Partner Sperm Motility ")
            partner_sperm_immunological_status = st.checkbox("Partner Sperm Immunological factors" )

            cervical_factors=1 if cervical_factors_status else 0
            female_factors=1 if female_factors_status else 0
            partner_sperm_concentration=1 if partner_sperm_concentration_status else 0
            partner_sperm_morphology=1 if partner_sperm_morphology_status else 0
            partner_sperm_motility=1 if partner_sperm_motility_status else 0
            partner_sperm_immunological=1 if partner_sperm_immunological_status else 0
            
        
        val_txtfile =[age_at_treatment, no_prev_treatment, no_prev_ivf_cycle, no_ivf_pregnancy,  female_primary, female_secondary, male_primary, male_secondary, couple_primary, couple_secondary, tubal_disease, ovulatory_disorder, male_factor,  patient_unexplained, endometriosis, cervical_factors,female_factors, partner_sperm_concentration, partner_sperm_morphology, partner_sperm_motility,   partner_sperm_immunological, embryos_transfer, total_embryo ]
        name_txtfile =["Patient Age at Treatment", "Total Number of Previous treatments, Both IVF and DI at clinic", "Total Number of Previous IVF cycles","Total number of IVF pregnancies","Type of Infertility - Female Primary",
    "Type of Infertility - Female Secondary", "Type of Infertility - Male Primary","Type of Infertility - Male Secondary","Type of Infertility -Couple Primary","Type of Infertility -Couple Secondary",
    "Cause  of Infertility - Tubal disease", "Cause of Infertility - Ovulatory Disorder","Cause of Infertility - Male Factor","Cause of Infertility - Patient Unexplained","Cause of Infertility - Endometriosis",
    "Cause of Infertility - Cervical factors","Cause of Infertility - Female Factors","Cause of Infertility - Partner Sperm Concentration","Cause of Infertility -  Partner Sperm Morphology","Causes of Infertility - Partner Sperm Motility","Cause of Infertility -  Partner Sperm Immunological factors",
    "Embryos Transfered","Total Embryos Created" ]

        
        

        if st.form_submit_button('Predict'):
            txtf = {"14":name_txtfile,"unnamed":val_txtfile, }
            txtfile = pd.DataFrame(txtf)
            
            output_txt, proba = main_predict(txtfile, model, threshold)
            #st.metric(label="Live-birth Occurrence Expectancy", value=proba, delta=output_txt, delta_color="off")
            st.metric(label="Live-birth Occurrence Expectancy", value=proba)

if __name__ == "__main__":
    run()