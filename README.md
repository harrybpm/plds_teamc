# A Simple Web App for Predicting Live-birth Occurence

## Background

- Infertility, among other factors, is considered the cause of unsuccessful conception. Advancements in medical procedure and physiology enable the fertilization process to occur outside of the human body, which is generally referred to as in-vitro fertilization (IVF). 

- IVF does not guarantee pregnancy even if the couple passes the assessment process. The factors that cause conception failure come in a broad spectrum, ranging from low to high significance. There could be some combination of factors or underlying factors that contribute to the successful conception in IVF, something that is missing if the assessment is made based on experience or incomplete statistical information. Moreover, IVF is well known to be a high-cost medical service with great uncertainty. 

- We aim to create a model that can predict live-birth occurrences based on patients’ medical records. The final model will be deployed as a web application, complete with a simple UI that shows prediction results.

## Objectives

- The main goal is to predict the live-birth occurrence based on the input given by the users

- The second goal is to give some consideration to the patients whether it is worth it or not to continue the IVF program. For example, the medical practitioner has a pessimistic hope, and the model predicts that the live-birth occurrence is low. The patients could stop the IVF program and use the money for other purposes

## Methods
![][method.png]

## About the dataset

## Dashboard
- The web app can be accessed here: https://ivf-livebirth-test.herokuapp.com/

- Simply fill in the form and press the predict button
![][dashboard.gif]

## Analysis
Prediction tends to give a negative result. This is probably due to an unbalanced dataset (i.e., more negative samples than positive samples). 

## Conclusion
- Deployed model in the web app could give live-birth occurrence prediction. Simple UI makes anyone could use the app without any problem.

- Though the app works as intended, the prediction result must not be used as the main reason when deciding to do IVF.

## References
- Goyal, A., Kuchana, M. & Ayyagari, K.P.R. Machine learning predicts live-birth occurrence before in-vitro fertilization treatment. Sci Rep 10, 20925 (2020). https://doi.org/10.1038/s41598-020-76928-z

- Ratna MB, Bhattacharya S, Abdulrahim B, McLernon DJ. A systematic review of the quality of clinical prediction models in in vitro fertilisation. Hum Reprod. 2020 Jan 1;35(1):100-116. doi: 10.1093/humrep/dez258. PMID: 31960915.

- https://github.com/FChmiel/ivf_embryo_prediction

