# Author: Vishal Kampani

![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/4ab4c14c-1400-4a44-91e5-14ef2295d108)
![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/19532b56-9c50-47cb-9cce-9ef5c0b1f678)
![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/9af33d36-6a84-4c36-9419-d75da266c7c7)

# Social Impact Vision
Create a product for the Providers, i.e., hospitals, medical professionals and consumers utilizing generative AI technology to enable cancer prognosis and diagnosis and help reduce the cost of care delivery and out of pocket cost for the **masses especially in developing and under-developed countries as there is an opportunity to impact SEVEN BILLION PEOPLE in these countries.**

# Motivation
There are numerous papers in last three years on using generative AI techniques in medical imaging. In 
fact, multiple papers claim to have State of the Art performance proven over one or few datasets.   However, there is limited open dialogue on commercialization of these state-of-the-art models.


Based on the literature search of papers with open-source code, it was difficult to find any paper whose code can be used as-is to solve these use cases, in some cases it was challenging to find open-source datasets that fit the needs of the code and in some cases the open-source datasets only had less than 100 patient data points, which is generally insufficient to train/test these models.

The majority of use cases are cancer focused as this has the most open-source datasets.  Also, the statistics that 1 in 3 people will have cancer, thus, prevalence is going to be high in the future with the aging population. Also, President Biden two years ago started Cancer Moonshot project (to prevent 4 Million cancer deaths by 2047), thus, there will be lot of federal funding in this area.

Below, I have outlined some of my thoughts on possible use cases.  

# Possible Real-life use cases 
## A)	Commercial Product Use cases (Most Preferred): 
### a.	Use Case#1: Clinical Decision Support 1 (Prioritization: Now): 
Input to model is either a 2D or a 3D whole body medical image and output is a segmentation of the organ and a simple classification of whether the person has cancer and the cancer grade classification including an image with the map of the cancerous cells along with a generative report describing the image findings. AI model will also prescribe a treatment plan.
### b.	Use Case#2: Clinical Decision Support 2 (Prioritization: Now):  
Input to model are medical images from time 0 (initial visit) and time 1 (follow-up visit 9 months later) and time 2 (follow-up visit 18 months later) and subsequent follow-up visits for patients who underwent radiation therapy/chemotherapy and output is a text output on whether the person has cancer and an image with the map of the cancerous region along with a generative report describing the image findings. AI model will also prescribe a treatment plan.
### c.	Use Case#3: Clinical Decision Support 3 (Prioritization: Next):  
Input to model are medical images from time 0 (initial visit) and time 1 (follow-up visit 9 months later) and time 2 (follow-up visit 18 months later) for patients who underwent radiation therapy/chemotherapy and output is a text output on whether the person has cancer and an image with the map of the cancerous region along with a generative report describing the image findings. The model will also predict whether in time 3 (follow-up visit at 24 months) cancer will spread in the lymph nodes or reach a new location in the body.  If it will spread, then an image will be produced showing the appropriate cancerous organ/nodes along with a generative report describing the image findings. AI model will also prescribe a treatment plan.
### d.	Use Case#4: Clinical Decision Support 4 – Consumer oriented wellness app (Prioritization: Future): 
Input to model is entire medical history. For patients who have had no cancer symptoms for 3 years or so, the wellness app will weekly monitor the symptoms and output will be whether they should seek physician advice based on the progression of their symptoms.  This app will be prescribed by physicians to patients in developing countries. 
**Job-to-be-done**: first-line medical professional (e.g., nurses, general physician, oncologist) in contact with patient needs a quick decision on whether person has cancer or is at risk of a relapse of cancer in future.  Also, medical professionals can learn only the most relevant to the patient case latest medical research from the tool.
**Problem statement**: In non-developed countries, the initial assessment of cancer is done by first line of staff, i.e., nurse practician or primary care general physician.  Over an 8-year clinical study, it was found that over 25,000 patients scanned, 60% of cases where a technologist said there was cancer, it was later found that there was no cancer. Hence, at the point of care when the patient is on the imaging table, the decision on whether the person has cancer can help avoid additional cost of further scanning and can then route the patient to the appropriate next stage of diagnosis.
**Social Impact**: This tool can fast-track diagnosis by 30%, especially in developing countries. Also, it can reduce wasteful spending of unnecessary diagnosis and lab tests in majority of cases, enabling millions of dollars in savings. Lastly, it gives cancer diagnosis abilities to medical professionals in remote parts of developing/under-developed countries, where the alternate is to not get treated at all.
**Model Outputs**: a) for current patient visit, text output of cancer present - yes or no; b) for current patient visit, image output of map of cancerous regions; c) for next patient visit, text output of cancer will be present - yes or no; b) for next patient visit, image output of map of cancerous regions (if applicable)
**Model Learning**: a) ability to distinguish cancerous regions from normal tissue and imaging artifacts; b) ability to apply NLP to radiologist and doctor’s notes to obtain relevant information; c) ability to predict whether cancer will appear/spread in the future.
**Model Training Dataset Needed**: for each patient need the following: a) X-rays; CT images and/or MRI images and radiologist reports from various visits; b) CBC (Complete Blood Count) and tumor markers blood and urine test; c) NGS (Next Gen Sequencing) DNA test; d) treatment drugs used; e) doctor’s notes on treatment plan and progression; f) tissue biopsy reports
**Key Features Needed in Dataset**: a) textual cancer type and TNM classification; b) segmentation of the cancerous region
**Data Formatting**: Preferably, the most common imaging parameters are used to acquire the MR, CT, x-rays. 
**Real world data considerations**: a) variation in textual reporting in doctor, radiologist and pathology reports; b) different imaging parameters are used for diagnostic images taken over several years.  
**Data source key user questions**:  what are the proportion of various cancer types in the training dataset? For each cancer type, what genetic mutations are included? What is the ethnicity, demographics distribution of the training dataset and does it represent the patient cases that are seen at my clinic?
**Data Privacy**: Initially, model will be deployed on-premise server and later as model matures it will be on cloud. Initially, patient data will be manually fed into the UX of the Clinical Decision Support tool. Later as model matures, it will be integrated with the electronic medical record, so patients’ history is automatically imported into the tool.
**Data Exclusion**: To get equal amounts of training data on all cancer types and all demographics, ethnicity is challenging. Thus, some patient groups and cancer types will be under-represented in training dataset. Hence, maintaining Fairness in dataset is difficult.
**Data Fragility**: In real world, obtaining training imaging data from different countries and different hospitals is challenging and training data needs to have images with different image acquisition parameters and different post-processing of images.

### B)	Use case: Education of physician and radiologist. 
Input to model is text entered of: TNM stage or location of tumor in an organ or grade of tumor. User selects which race the model should output.  The model outputs a generative medical image with tumor of the selected race based on the input cancer characteristics.
Job-to-be-done: educate physician on imaging characteristics nuances specific to the race of the patient they are seeing.   
Problem statement: The existing radiological training material was created by developed countries, thus the patient data used to educate the radiologist is predominantly from the white race.  The population that a radiologist would see in a developing country is predominantly not from the white race.  Thus, there is a need to create training material for the types of patients that a physician would see in their country.
Impact: Provides education access to many physicians in developing and underdeveloped countries at much lower cost compared to the costly radiology training in universities of developed world.

C)	Use case: New Drug Application: Input to model is a white race patient cancer medical image (before new drug intervention) and a medical image (after new drug intervention). Output of model is for any race selected by researcher; model should output both medical images (before new drug intervention and after new drug intervention).
Job-to-be-done: Life science companies have a need to provide diverse race drug data efficacy in support of their new drug application.  
Problem statement: FDA as part of new drug approval application is asking for more diversity data from clinical trials. However, there is an under-representation in participation in clinical trials of non-whites.  Hence, the efficacy data submitted is skewed for the white race.
Impact: The cost of one patient enrolled in a clinical trial is $50K.  Phase 3 trial needs hundreds of patients enrolled.  Getting access to and convincing non-white patients to enroll in trials is challenging. Thus, the proposed AI model will provide more diverse patients results and possibly faster FDA new drug application approval timelines.

D)	Use case: Synthetic images. Input to model is one medical image and output of model is multiple augmented images that represent real patient scenarios.
Job-to-be-done: to effectively train AI models hundreds and thousands of medical images representing diverse set of patient disease characteristics are necessary. 
Problem statement: it has been challenging to confirm whether a generative model represents the real-life situation of an actual patient.  More diverse data sets fed as input to model, will produce output that represents actual patient disease characteristics.
Impact: if we are able to show that generative model images output represents real-life patient scenarios, then there would be mass adoption globally of these AI models.

E)	Use cases: 
a.	Use case#1: Automated treatment plans- Clinical Decision Support: Input to model is any available medical images, genetic testing results, patient history, lab results. Output is a recommendation on the treatment plan for the oncologist. 
b.	Use case#2: Optimize Number of Imaging and Diagnostic Test – Clinical Decision Support: Input to model is any available medical images, genetic testing results, patient history, lab results. Output is a recommendation on next set of lab results or imaging tests to prescribe.
Job-to-be-done: supplements knowledge of entry-level physician or oncologist who may not have the many years’ experience or knowledge of constantly changing cancer guidelines. The automated plan output increases confidence of treating oncologist.
Problem statement: oncologist in developing country may not have the time (due to high case load) and money to continuously update themselves on changing cancer guidelines and clinical trials and knowledge resources. For every patient, a standard set of lab tests and diagnostic images are prescribed, many of these tests are unnecessary and are done for insurance purposes or to build confidence in treatment plan.
Impact: the time save by an oncologist in a developed or developing country from learning about changes in the cancer field, that time can be spent on caring for patients, increasing patient satisfaction and staff satisfaction at the Provider. 2X additional time can be spent in face-to-face patient interactions. Also, an optimized recommendation on diagnostic and lab tests, save wasteful money spent in the order of hundreds of millions of dollars. 

F)	Use case: Gene Mutation Prediction - Clinical Decision support: Input to model is any available medical images, patient history, lab results. Output of model is which genetic mutation patient has.
Job-to-be-done: oncologist needs knowledge of genetic mutation patient has to prescribe a personalized treatment plan
Problem statement: genetic mutation testing for a panel of test can cost $500 to $2000 per patient. For most developing countries this is a big cost that is paid out of pocket by the patient. Also, several countries don’t have the infrastructure, i.e. genetic testing machines which can cost $500M, and maintenance cost of $50K/year.  The high cost barriers limits the number of patients who have their genome sequenced.
Impact: it takes 8 weeks for genomic testing results to be published to physician. Eliminating the need for genetic testing will be transformational in the healthcare industry. 

G)	Use case: Patents: Given an invention disclosure and claims as input to the model. Model outputs a ranking of top 10 most similar patents claims.  Can be used by Companies and Patent Office in searching for similar patents. 
Job to be done: need to search for similar patents as the invention being submitted.
Problem statement: patent submission form requires that inventor has performed due diligence that his invention doesn’t infringe on the rights of existing patents and the claims have not been disclosed globally in any format prior to the priority date. Patent office takes a long time (1 to 2 years) to process a patent application as they have to perform a comprehensive search on similar patents.  Also, if a patent goes to the courts, then the lawyers spend a lot of their time doing these searches too.
Impact: Will speed-up the patent approval process by 80%, reduce number of patent disputes that goes to courts by 30%. 

H)	Use case: Personalized treatment planning across genetic pools – Clinical Decision Support: Input to model is genetic characteristics of a particular race or people from a developed country and genetic characteristics of under-developed country. Output of model is personalized treatment plan for patients in under-developed country. Thus, literature and clinical trial advances of developed country are translated to the patients in under-developed country.  The AI model learns characteristics of patient populations in both of these countries.
Job to be done: oncologist in under-developed countries need to prescribe a personalized genetic based treatment plan
Problem statement: there is a lack of literature and clinical trials and understanding of cancer pathways in patients of under-developed countries. 
Impact: Billions of people in developing and under-developed countries can benefit from the advance knowledge of treatment plans in developed countries. 
