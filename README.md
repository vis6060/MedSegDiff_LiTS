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

### Use Case#1 to #4 Analysis

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

***Job-to-be-done***: educate physician on imaging characteristics nuances specific to the race of the patient they are seeing.   

***Problem statement***: The existing radiological training material was created by developed countries, thus the patient data used to educate the radiologist is predominantly from the white race.  The population that a radiologist would see in a developing country is predominantly not from the white race.  Thus, there is a need to create training material for the types of patients that a physician would see in their country.

***Impact***: Provides education access to many physicians in developing and underdeveloped countries at much lower cost compared to the costly radiology training in universities of developed world.

### C)	Use case: New Drug Application 
Input to model is a white race patient cancer medical image (before new drug intervention) and a medical image (after new drug intervention). Output of model is for any race selected by researcher; model should output both medical images (before new drug intervention and after new drug intervention).

***Job-to-be-done***: Life science companies have a need to provide diverse race drug data efficacy in support of their new drug application. 

***Problem statement***: FDA as part of new drug approval application is asking for more diversity data from clinical trials. However, there is an under-representation in participation in clinical trials of non-whites.  Hence, the efficacy data submitted is skewed for the white race.

***Impact***: The cost of one patient enrolled in a clinical trial is $50K.  Phase 3 trial needs hundreds of patients enrolled.  Getting access to and convincing non-white patients to enroll in trials is challenging. Thus, the proposed AI model will provide more diverse patients results and possibly faster FDA new drug application approval timelines.

### D)	Use case: Synthetic images. 
Input to model is one medical image and output of model is multiple augmented images that represent real patient scenarios.

***Job-to-be-done***: to effectively train AI models hundreds and thousands of medical images representing diverse set of patient disease characteristics are necessary. 

***Problem statement***: it has been challenging to confirm whether a generative model represents the real-life situation of an actual patient.  More diverse data sets fed as input to model, will produce output that represents actual patient disease characteristics.

***Impact***: if we are able to show that generative model images output represents real-life patient scenarios, then there would be mass adoption globally of these AI models.

#### E)	Use cases: Optimize Number of Imaging and Diagnostic Test – Clinical Decision Support: 
Input to model is any available medical images, genetic testing results, patient history, lab results. Output is a recommendation on next set of lab results or imaging tests to prescribe.

***Job-to-be-done***: supplements knowledge of entry-level physician or oncologist who may not have the many years’ experience or knowledge of constantly changing cancer guidelines. The automated plan output increases confidence of treating oncologist.

***Problem statement***: oncologist in developing country may not have the time (due to high case load) and money to continuously update themselves on changing cancer guidelines and clinical trials and knowledge resources. For every patient, a standard set of lab tests and diagnostic images are prescribed, many of these tests are unnecessary and are done for insurance purposes or to build confidence in treatment plan.

***Impact***: the time save by an oncologist in a developed or developing country from learning about changes in the cancer field, that time can be spent on caring for patients, increasing patient satisfaction and staff satisfaction at the Provider. 2X additional time can be spent in face-to-face patient interactions. Also, an optimized recommendation on diagnostic and lab tests, save wasteful money spent in the order of hundreds of millions of dollars. 

#### F)	Use case: Gene Mutation Prediction - Clinical Decision support: 
Input to model is any available medical images, patient history, lab results. Output of model is which genetic mutation patient has.

***Job-to-be-done***: oncologist needs knowledge of genetic mutation patient has to prescribe a personalized treatment plan

***Problem statement***: genetic mutation testing for a panel of test can cost $500 to $2000 per patient. For most developing countries this is a big cost that is paid out of pocket by the patient. Also, several countries don’t have the infrastructure, i.e. genetic testing machines which can cost $500M, and maintenance cost of $50K/year.  The high cost barriers limits the number of patients who have their genome sequenced.

***Impact***: it takes 8 weeks for genomic testing results to be published to physician. Eliminating the need for genetic testing will be transformational in the healthcare industry. 

#### G)	Use case: Patents: 
Given an invention disclosure and claims as input to the model. Model outputs a ranking of top 10 most similar patents claims.  Can be used by Companies and Patent Office in searching for similar patents. 

***Job to be done***: need to search for similar patents as the invention being submitted.

***Problem statement***: patent submission form requires that inventor has performed due diligence that his invention doesn’t infringe on the rights of existing patents and the claims have not been disclosed globally in any format prior to the priority date. Patent office takes a long time (1 to 2 years) to process a patent application as they have to perform a comprehensive search on similar patents.  Also, if a patent goes to the courts, then the lawyers spend a lot of their time doing these searches too.

***Impact***: Will speed-up the patent approval process by 80%, reduce number of patent disputes that goes to courts by 30%. 

#### H)	Use case: Personalized treatment planning across genetic pools – Clinical Decision Support: 
Input to model is genetic characteristics of a particular race or people from a developed country and genetic characteristics of under-developed country. Output of model is personalized treatment plan for patients in under-developed country. Thus, literature and clinical trial advances of developed country are translated to the patients in under-developed country.  The AI model learns characteristics of patient populations in both of these countries.

***Job to be done***: oncologist in under-developed countries need to prescribe a personalized genetic based treatment plan

***Problem statement***: there is a lack of literature and clinical trials and understanding of cancer pathways in patients of under-developed countries. 

***Impact***: Billions of people in developing and under-developed countries can benefit from the advance knowledge of treatment plans in developed countries. 

# Literature survey
Please see the Deep Learning Journal Articles Survey.xlsx file.

# Open-source datasets
Here are the open source datasets or websites that host a library of cancer datasets that I came across:

  a)	Chest x-rays:
  
    a.	CheExpert dataset (Stanford AIMI - https://stanfordmlgroup.github.io/competitions/chexpert/)
    
    b.	https://physionet.org/
    
    c.	MIMIC CXR dataset (https://www.citiprogram.org/members/index.cfm?pageID=122&intStageID=106240#view)
    
  b)	MR:
    a.	BraTS dataset (https://ipp.cbica.upenn.edu/)
    b.	CT & MR dataset for 15 organs (https://amos22.grand-challenge.org/Instructions/)
    
  c)	CT
    a.	Liver tumor segmentation (https://competitions.codalab.org/competitions/17094#learn_the_details)
    
  d)	Websites that host library of many datasets:
    a.	https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890

# Journal Paper Explored & Sample size
The codebase and methodology presented in the following paper was used: MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer. This paper show State of the Art (SOA) performance for the AMOS2022 dataset of 200 multi-contrast CT images.  

***It was encouraging to see a code that was able to generate SOA performance with meager 200 images.  Thus, to further validate the performance of the model on different datasets, I trained the model on the dataset described in the - The Liver Tumor Segmentation Benchmark (LiTS) journal article, which is an open-source dataset of 110 3D CT images.***

# MedSegDiff Code modifications
The code from, https://github.com/WuJunde/MedSegDiff, was modified to fit the requirements of LiTS dataset and some other considerations as follows:

  a)	Since, a single GeForce RTX 4090 was available for training, the parallel training code was commented-out in the file – train_util.py.
  
  b)	Since, windows desktop was used, the gloo backend option was inserted in the file – dist_util.py.
  
  c)	The parser in bratslaoader.py 3D data section of the code was updated to match the format of the input data folders name.
  
  d)	The actual code is for one 3D MR image which when converted are 155 2D DICOM images.  However, the one 3D CT image of the LiTS dataset has 123 2D image. Thus, all the lines in bratsloader.py that dealt with 155 numeric calculations were changed to 123.
  
  e)	The initial code for each patient they had four MRI series – T1; T1Flair; T2; post-contrast T1 weighted. Hence, the number of channels in bratsloader.py were set at five in original code.  In my case for the LiTs dataset there is only one CT image per patient, hence number of channels was set at two.
  
  f)	The initial code has hard-coded the flag in the bratsloader.py data file between if command which has two files in training code and one file (as there is no segmentation file) in testing code.

# Results
The model was trained with 110 3D CT images of the LiTS dataset. It was trained for 100,000 steps using the following hyperparameters after pytorch environment activation:

  •	conda activate pytorch-gpu2-python-3-10
  
  •	python scripts/segmentation_train.py --data_dir data/training --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8
  
  •	python scripts/segmentation_sample.py --data_dir /data/testing --model_path results/savedmodel100000.pt --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 50 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5  --dpm_solver True

In the testing command, changes were made to hyperparameters of diffusion steps by changing it from 50 to 10,000. Further in a different scenario, the number of ensembles were changed from 5 to 20.  However, no difference was observed in the output.

Below is an example of patient#16 3D CT image of the three views – A/P; S/I and R/L directions (Fig 1) and the liver segmented mask (Fig 2). Lastly, there is the model output segmentation mask (Fig 3).

Even though the original code was tested on 3D MR images, it also works for 3D CT images.

Fig 1: Patient#16: Training 3D CT image
![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/a479d11c-2d7a-4565-8efc-ccdc7db27c77)

Fig 2: Patient#16 LiTS Testing dataset 3D CT Segmented image (Reference)
![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/0eb67b43-9edb-4cef-862b-19f352848334)

Fig3: Patient#16:Model Output Segmented image (Actual)
![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/0ebd23b4-b780-4ac1-bb83-22a03279fb78)

***It can be seen that the model output segmented mask does not match up with the desired segmented mask.*** This is possibly due to limited training data sample size in the LiTS dataset.  Based upon datasets used in the “MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer” 1200 to 8046 images. 

# Inferencing using Google Cloud – Vertex AI and Storage Buckets

The following is my experience of using the Jupyter Lab notebook and Jupyter terminal in Vertex AI with the PyTorch code:

a) Vertex AI User documentation is very developer focused as compared to for a novice. Each documentation page has links to many other pages, so it is difficult to keep track of the relevant information needed to solve my particular issue at hand. The user documentation needs more tutorials showcasing how to execute a particular workflow of steps. The amount of information on each user documentation is overwhelming. After reading many of the sections, I was still unsure what series of steps I need to take to execute on my particular use case.

b) Troubleshooting help: In some cases, there are several blogs of people having same error, but few have replied a resolution. When I tried those resolution steps it did not work for me.

b) Folder Upload to Storage Bucket: On selecting a folder from local computer it would not upload to Google cloud and offers no error message as to why it did not upload.

c) GPU Selection: It took an hour or so, just to find a region which has a machine with Python and CUDA along with a GPU.  I had to go one-by-one each server region to find the one that would spin-up a virtual machine with the GPU. Some visual guidance in the UI as to which region are only available with this setting would have made this process easier.  Finally, it was the West3B region that worked for me.

d) Billing is not real-time. after 2hr of shutting down my instance found out there was more charge i had accrued.  Also, GPU usage can be expensive, I had to pay $8 for about 8 hours of Vertex AI machine time of the cheapest GPU available, I can see the cost scaling if I have to troubleshoot for several hours/days and perform inference on 100s of 3D CT images.  Lastly, after I subscribed for the 90-day free credits, I found out that free credits do not allow use of GPUs.  I later found that this was written in the description of free credits, which I initially did not see.

e) File Upload: When i upload into the instance Jupyter lab notebook, it automatically compresses the size of the file from 500MB to 240MB, not sure if the contents of the compressed file are same as original file. But the same file when uploaded into Google cloud bucket the size stays the same.

f) File read error: Vertex AI was unable to read the NII file extension CT testing images. Even after installing the NiBabel python library. It is an issue with Google Cloud Vertex AI as the exact same code and testing image was read on my local computer.

g) Bard and ChatGPT3.5 both gave many suggestions on the possible root cause to fix the error of unable to read NII files.  I followed some of the suggestions, but still could not fix the error.  They also gave code that I could insert into my files, but this code did not solve the error. 

# Challenges

a)	The single GeForce RTX 4090 to train 100 3D CT images took 8 days * 24 hours.  However, for testing I was able to inference on only two images at a time, more than 2 images I use to get out of memory error message.

b)	Bard and ChatGPT suggestions were of minimal help to troubleshoot error messages.

c)	Google Vertex AI Jupyter terminal was unable to read 3D CT Images in the NII file formats.

d)	Open-source AI codes for the desired application are limited. There are only a few journal papers that have made their codebase public.  Further, for these free codebases the authors have posted a working code, but there are several implementation issues when others try to execute the same code with a different dataset on their computer.  

# Summary/Conclusion

Several commercially viable use cases were introduced here.  However, it was a challenge to find open-source datasets and the corresponding open-source codebases that can be utilized to create a prototype AI model, which can be used for initial customer feedback. For example, an open-source dataset for CT images cancer liver segmentation was used on an open-source codebase, initially developed by OpenAI, which showed state-of-the-art results for MR images. But for the CT images given the limited dataset of approximately 120 images the training model was poor.  

# Future Direction

a)	Need an easier way to string the various public codebases, each with a different application, together, so a commercially viable use case can be introduced to the market.

b)	Various open-source codes are based on the few big open-source datasets mainly of X-ray. Big open-source datasets of CT and MR imaging are lacking.  Hence, large healthcare institutions with the appropriate patient consent with a large historical dataset are well poised to develop AI models.  Sometimes individual hospitals datasets may not have the needed equal number of datapoints patient diversity, thus federated mechanisms of pooling datasets across healthcare institution are required.

c)	Given that LiTS dataset is for only 110 3D CT images. We need to introduce further data augmentation techniques or insert synthetic images which represents the real patients into the training dataset.  But these augmentation techniques and synthetic images need to represent real patient cases.

d)	Industry needs to figure out a way to reduce the barriers to access to datasets by making them cheaper or open-source so as to reduce the cost of development of these AI models as the revenue from these products may be too little compared to the cost.

# References

https://github.com/WuJunde/MedSegDiff

https://github.com/openai/improved-diffusion

# IDEO Framework
## IDEO Theory of Change for Use Case A:

The IDEO theory of change is used to brainstorm the changes that will be needed to achieve the desired outcomes.
(https://www.designkit.org/methods/explore-your-theory-of-change.html)

### Change #1: 
***From***: no concept of personalized prognosis of cancer even possible in developing countries. Majority visit a specialist or cancer hospital when they have severe symptoms.

***To***: spreading awareness of availability of prognosis tools.

### Change#2: 
***From***: lack of any prognosis tools both in developed and developing countries. 

***To***: The generative AI tools empower the third-tier district/village/small-town level non-oncologist doctors or oncologist to perform prognosis.

### Change#3: 
***From***: after the cancerous cells have been eliminated, still there is uncertainty on whether cancer will appear at the same site or at a different site in the body. Both in developed and developing countries.

***To***: a change of follow-up visits schedule from every 6 to 9 months, to skipping some visits to eliminating all follow-up visits, as prognosis of possibility of recurrent cancer can be done at third-tier hospitals. 

### Change#4: 
***From***: patients having to see a physician every 6 to 9 months, even if they have been cancer-free for several years. 

***To***: patients use a wellness app to determine if their symptoms show a relapse of cancer.

### Change#5: 

***From***: oncologist in developing countries have a heavy workload, and don’t have the time to stay up to date on latest advances/research.

***To***: oncologist relying on the Clinical Decision Support tool to provide them with just the relevant information necessary for the diagnosis of the particular patient using the Explainable AI methodology.

## IDEO Ecosystem Mapping: 

Patients as the user: The behavior or outcome you want to help your user achieve: A behavior change of weekly logging-in and using the wellness app.  Have confidence in the results of the app, which is going to be a simple message that is shown – “Please report your symptoms to your doctor.”  The patient needs to be trained on how to use the app.  The patient needs to be educated with the possibilities of cancer occurrence.

Relevant Community Personas around the user/patient: Family, Relatives, Cancer Support Groups, Healthcare Delivery Organizations.  Each of these groups would be a fan of the Clinical Decision Support app and will support he user.  Education and Awareness of the Clinical Decision Support app would be required among these personas.

Services user needs:  A person to educate the user on the app. Low degree of challenge as their family members or healthcare delivery organization can provide the education.

Institutions – rights and freedom of the user: timely access to healthcare is a right that the patient ought to get. But, in several developing countries access to healthcare requires an upfront payment, and thus is a barrier. Thus, the app when it gives the message to consult a healthcare professional based upon their symptoms, this may panic the individual if they are unable to seek timely help. 

# Google’s Responsible AI Framework

Referenced from here: https://ai.google/responsibility/responsible-ai-practices/

## Is AI the right solution for this customer problem?

For the Cancer CDS tool, AI probably better to solve user problem in a unique way due to several reasons:

***a)	The core experience requires recommending different content to different users***: in this case, depending on the different patient’s health conditions different treatment plan needs to be recommended.

***b)	The core experience requires prediction of future events***: clinical decision support model needs to predict future spread of cancer in body

***c)	Personalization will improve the user experience***: physician user experience is improved when the local treatment guidelines, i.e. personalized to that healthcare institution, is offered as explanation for recommendation along with the national consensus treatment guidelines.

***d)	User experience requires natural language interactions***: Clinical Decision Support model requires NLP abilities for the physician and radiologist report findings. 

***e)	The most valuable part of the core experience is its predictability in-context***: prediction needs to take into account the particular condition of the patient.

***f)	The cost of errors is low***: as this is only a clinical decision support tool meant to provide guidance to the medical professional.

***g)	Augmentation is required***: patient’s history can be long and could take a lot of time to evaluate it, thus, the AI model reviewing the entire history and offering suggestion is preferred. The stakes of getting the diagnosis/prognosis wrong is high from a patient safety and resources standpoint.

## Reward function: 

***AI model will be optimized for recall, i.e. fewer false negatives offers the most user benefit***, because if a patient will develop cancer symptoms and if the model says that the person will not have cancer, then that ***puts the patient at risk, it decreases the confidence of the medical professional to trust the clinical decision support model***. Sacrificing precision in favor of recall means that there will be several instances when the model will incorrectly say the person will develop cancer in reality they will not, this would increase the workload of the clinician and if this continues then over a period of time, ***they may use less of the clinical decision support model.*** 

## Success metrics: 

A)	Ideal scenario is that the model is being used by clinician on 70% of their patients. ***If it is being used on less than 70% of the patients in a month***, then alert the design team to conduct a user research interview with 5 why’s or fishbone analysis to determine root cause. 

B)	Ideal scenario is that model offers less than 10% false negatives, ***if the in-model instant user feedback shows that more than 10% patient cases the model said patient doesn’t have cancer, but actually had cancer***, then alert the design team so they can conduct a root cause analysis.  

These metrics will change after 10,000 patients have been processed and the model has been revised from this feedback. When the model matures, the expectation is that metric A is revised where it is used for 90% of patients and metric B is revised where the model offers less than 2% false negatives.

## Mental models:

### Section1: User groups:

  a)	Oncologist in developed countries
  
    Primary goal: Serve a large workload of cancer patients and stay up to date on latest cancer drugs and treatment pathways.
    
    Step-by-step process that users currently use to accomplish the task that AI will accomplish: Currently, there is no tool that will alert the oncologist whether cancer has spread to parts of body for the various ethnicities and genetics.
    
      i. They manually review the entire medical history of patient, including numerous pages of radiologist notes and past treatment history. 

      ii.	To determine if cancer has spread or might spread to other parts of body, they perform CT or MRI scans and blood tests at every 9-month patient visit.

b)	Oncologist in under-developed countries

  	Primary goal: offer a low-cost option to patients by being knowledgeable enough to serve cancer patients, as access to oncologist is difficult. Also, patients access to generic cancer drugs is limited and there is insufficient information the efficacy of any given drug for their ethnic population.
   
	Step-by-step process that users currently use to accomplish the task that AI will accomplish: Currently, there is no tool that will alert the oncologist whether cancer has spread to parts of body for the various genetics.
 
    i.	They manually review the entire medical history of patient, including numerous pages of radiologist notes and past treatment history. 
    ii.	No follow-up visits from patients due to resource challenges. Thus, patients are treated only once they once again show severe symptoms.
    iii.	They rely on memory of past similar patients cancer cases, to prescribe a treatment plan for the current patient case.

c)	Nurse practician or primary care general physician in developing and under-developed countries

	Primary goal: offer a low-cost option to patients by being knowledgeable enough to serve cancer patients, as access to oncologist is difficult. Also, patients access to generic cancer drugs is limited and there is insufficient information the efficacy of any given drug for their ethnic population.
   
	Step-by-step process that users currently use to accomplish the task that AI will accomplish: Currently, there is no tool that will alert the oncologist whether cancer has spread to parts of body for the various genetics.

    i.	They manually review the entire medical history of patient, including numerous pages of radiologist notes and past treatment history. 
    ii.	No follow-up visits from patients due to resource challenges. Thus, patients are treated only once they once again show severe symptoms.
    iii.	Since, they are not trained oncologist and lack recent, they mostly prescribe the same treatment pathway to all patients, irrespective of patient condition nuances – especially due to limited availability of expensive blood test assay; diagnostic imaging technology; forced to prescribe non-cancer drugs due to limited availability of any cancer drugs.

### Section2: Potential places where user’s mental model could break when encountering realities of AI functionality: 

i.	Some patient’s medical history might be comprehensive and over many years and for some patients medical history might be short as past records are unavailable, but the AI model output could be same for a patient from these groups, thereby confusing the medical professional. There would be confusion as to which medical history parameter makes the difference.

ii.	The model output may be the same for a long radiologist report or from the summary taken from the report. Thereby confusing the user if the AI model is taking into account the nuances of the entire radiologist report findings.

### Section3:  what cause and effect relationships does the user need to understand —even in simplified terms or by analogy — to successfully use the AI product?

At different time points of the follow-up visits, the diagnostic images can be of different organs coverage and could be acquired with different parameters. The AI model will assume that all the images are consistently acquired with the same parameters. 

### Section 4: how might anthropomorphizing the product alter the mental model?

Model can be made more human-like, when the output of the model is accompanied by an explanation of all the model inputs that were considered and not considered.  A medical professional evaluating the relevance of a patient’s symptoms and medical history would perform similar explanations of what is relevant and irrelevant t the situation at hand.

### Section 5:  The biggest risks to users developing good mental models for our product are:

Since the model can offer an explanation on which of the medical history parameters are relevant and not, there may be an inclination to order more lab tests and more diagnostic exams during follow-up visits, just to check the boxes of the various types of data that needs to be entered to the model. Practices would do this to reduce their malpractice liability as the AI model explanations can be used as justification for their actions.

### Section 6: Onboarding: 
The Clinical Decision Support product will provide user guidance on whether a person based on their medical history has cancer, it will segment the organ(s) that are cancerous, it will provide prediction on the body parts that the cancer can spread. Benefits to user are time savings, a “second opinion” tool and delegation of follow-up cancer monitoring to non-oncologist medical professionals. Its primary limitations are that initially, there is higher likelihood that model output is incorrect. i.e. model says the person has a particular grade of cancer, when that is incorrect or it might see something cancerous in diagnostic images, when that could be just an artifact.  Hence, it requires your help to flag and provide an explanation when the model output is correct and incorrect. This feedback will be reviewed by human reviewers and used to improve the model performance.    

***In boarding messaging***: provide a short video demo of how to upload the various pieces of the medical history into the model and the output of model.  There will be no real-time updates to the model, all the feedback will be accumulated and then in consultation with the user the model will be revised.

### Section 7: Feedback + Control
***Feedback Mission***: To collect whether the model accurately able to provide diagnosis and prognosis output.  Explicit feedback of thumbs up/down will directly be used for model tuning, rest of the qualitative feedback will be used by design team to better understand model performance in real-world across diverse patients/users.

***Degree of User Control over Model***: User has full control over the system, as it is a clinical decision support tool and the stakes are high as it deals with a disease that can be fatal, input patient medical history data will be automatically loaded into the inferencing model, but it is up to the medical professional to heed the output of the model or do it the status quo way.  This way as the user builds trust with the model, they will heed model’s output more often for more patients.

***Explicit Feedback***: survey questions, thumbs up/down; open text field are going to be in-context for each patient visit.

***User Motivation***: For the initial deployments of the model at customer sites, offer the site 2X time it takes for the different user personas to provide feedback at the hourly rate.  Because, initially they are doing the model a favor by helping it improve. For example, if takes 15minutes/patient to provide feedback and the blended hourly rate for all the personas that need to complete the survey is $100, then pay the clinic $50/patient survey complete.  After a large number say, 10,000 patients of model tuning and success metrics satisfied, then get only thumbs up/down feedback, and rely on the intrinsic social motivation for user to provide this feedback.

***Opt-out***: Allow users to opt-out from sharing their implicit behavior of their users with the company in the terms of service.

### Section 8: Explainability + Trust

This Clinical Decision product has high user impact as chance of error can be costly and fatal for a patient.  Hence in the 2x2 of User Impact and AI Confidence, this solution is in the red circle region. In this region, to develop user trust need to provide explanations both overall model explanation and general and specific output.  

The overall model explanation will describe the training dataset and what output to expect and not to expect as this is intended for technical audience. Also, aggregate visualization of the training dataset will be shown, for example, a pie chart that shows the different types of cancer the model was trained on.  For general output, it will not provide a percentage of AI confidence level, instead it will provide an explanation of what it means for a categorical label to be high confidence, medium confidence, or low confidence.  For specific output, it will provide top 3 recommendations along with the same high, medium, low confidence labels.  For a future release, AI model will say which specific aspects of the medical record were similar to data in the training dataset.  It cannot offer example-based justification from the dataset, due to possible plagiarism of the dataset and to maintain the privacy of the patient dataset.

 ![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/eec3c395-efca-40d7-8c95-dc05eb47cf35)

### Section 9: Errors

The following errors are possible:

a.	Context error: incomplete coverage of the mapping of current cancerous area and incorrect mapping of the organ for the particular patient. 

b.	Context error: prediction of a cancerous region can be completely wrong or partially wrong for the particular patient.

c.	Context error: if one follow-up visits image is unavailable, but others are available, then user may not use the system, even though the system can work just fine with its prediction with the missing image, for the particular patient.

d.	Fail State error: insufficient training data for the particular case at hand. Hence, low confidence recommendation produced.

e.	Background errors: system produces an incorrect output and the user, due to lack of time or poor judgment, is unable to discern the incorrect output.

f.	Input errors: unexpected input: different formats of patient records. Some missing data points.  User will think system will auto-correct or fill-in the gaps when it provides its output.

# Product Requirements Document (PRD) & Product Strategy

## Problem Statement
Currently, there is no cancer prognosis tool in the market. AI models are best suited to address this problem. Physicians have to perform painful and costly tissue biopsy studies to determine if cancer has spread to parts of the body.  Physicians are involved only when cancer has reached stage I/II/III/IV, there is no way for physicians to assign a likelihood to patients that cancer will happen based on their symptoms. Also, insurance companies are burdened with unexpected high cost when a patient is diagnosed.  Further, patients don’t have a way to continuously monitor whether the daily symptoms they are experiencing it means that cancer has relapsed, thus, they are in everyday stress thinking about it which affects their mental well-being and quality of life.  

## Cancer Clinical Decision Support (CDS) tool Use Case- Product Vision

Goal is to be the first in the industry to develop an AI-powered cancer diagnosis and prognosis tool across at least five cancer types with a unique curated dataset that is accurate, explainable, trustworthy, fair, and reliable. This tool will be used by providers in all parts of the world, with a particular focus on underserved communities. It will improve patient outcomes, reduce healthcare costs, and empower patients to take control of their health. We will achieve this by building a sustainable business model that ensures that everyone has access to this life-saving technology.

## Mantra

“Become the first team to successfully commercialize a cancer prognosis AI clinical decision support tool globally.”

## Objectives

  1.	Integrate the tool into the workflow of at least 35% of Providers in the US. These are all paying customers. The tool is being used to generate accurate and timely prognoses and diagnoses for at least 50% of patients monthly by these customers.
     
  2.	Integrate the tool into the workflow of at least 30% of Providers of Canada, UK, Germany, Italy, and France.  These are all paying customers.  The tool is being used to generate accurate and timely prognoses and diagnoses for at least 50% of patients monthly by these customers.
     
  3.	Adoption of a cancer wellness app by at least 60% of patients who are diagnosed to be cancer-free, which will help them to maintain their health and reduce the risk of recurrence.
     
  4.	Adoption of the CDS tool by at least 25% of health insurance companies in the US.  These are paying customers.
     
  5.	Integrate the tool into the workflow of 10% of Providers in five developing countries and under-developed countries with a particular focus on countries with high rates of cancer incidence. The tool is being actively used on at least 60% of patients monthly by these customers.

## Impact

This tool can fast-track diagnosis by 30%, especially in developing countries. Also, it can reduce wasteful spending of unnecessary diagnosis and lab tests in majority of cases, enabling millions of dollars in savings. Lastly, it gives cancer diagnosis abilities to medical professionals in remote parts of developing/under-developed countries, where the alternate is to not get treated at all.

## Constraints 

•	The large sample size of 100,000 to 1,000,000+ of patients in selected countries with different cancer types and multimodal data of: genetics, CT/MR imaging at different times of their follow-up visit, lab tests and symptoms of a cancer-free patient, possibly does not exist.  Hence, quite expensive and time-consuming prospective studies will need to be initiated.

•	These multimodal data for each patient is 300MB+, thus, state-of-the art multiple expensive NVIDIA H100 GPUs will be needed, possibly along with Google or AWS cloud cost.

•	This will be first-of-its-kind AI model, most likely developed from scratch, hence validating its output will take several alpha customer sites and several months of revising the model. Training the model will also take several months of trial and error. Explainability of the model is a key aspect, and extensive research will be needed to provide this ability that meets user expectation. It will take several months to get the model FDA approved too.

•	It is unclear whether anybody will pay for this tool and if so, how much is their willingness-to-pay.

•	The use of AI Clinical Decision Support technology is new; thus, their will have to a change in user personas and customer mindset to convince them to adopt this technology.

## Personas

•	Medical Oncologist & Radiologist at Providers in Developed and Developing countries

•	General Physicians & Nurse Practicians in Developing and Under-developed countries

•	Healthcare Insurance Agents in the US

## Market Insights

According to Markets and Markets report, the AI in Healthcare Market is projected for big growth from USD 14.6 Billion in 2023 to USD 102.7 Billion by 2028, at an impressive CAGR of 47%.

Also, an aging population living longer means that there will be more people getting cancer in their lifetime and hence more demand for a Clinical Decision Support tool.  It is estimated that 1 in 2 women and 1 in 3 men in the US will get cancer in their lifetime. (https://www.medicalnewstoday.com/articles/288916)

Further, there is a shortage of skilled medical professionals in developed countries and more so in developing and under-developed countries.  

Lastly, the spread of internet and high adoption of mobile smartphones in developing and under-developed countries means more awareness for people to seek preventive cancer tests, and hence more need for an automated point-of-care Clinical Decision Support tool.

## Business Model Canvas
![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/df52a7ab-d1bb-43a4-8507-a4bb07e355e8)

## CDS Tool Jobs-to-be-Done
![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/d3e881b0-7d61-41ac-93c1-df042fb4d80e)

![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/a88bc5eb-b57c-4c8f-8835-0e5c34828be5)

![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/64ce3f15-1236-4bb9-84e7-8cd5c35e270f)

![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/19fc5281-41be-467c-be92-9817915e4d6d)

## Use Cases Scenarios

A Visual Patient journey map is in the appendix.

***Scenario 1***: In the US, a patient who has never had cancer, but has some symptoms visits their general physician, some lab tests are ordered.  Based on the lab results, the CDS tool assists the physician in determining whether patient has cancer or is likely to in the future to develop cancer.  If so, the patient is referred to the medical oncologist, who orders diagnostic imaging and lab tests and Next-Generation Sequencing DNA tests.  After the completion of the imaging, an automated report is generated with the findings to be reviewed by the radiologist.  The oncologist based on all this multi-modal data, uses the CDS tool to determine whether person has cancer and which grade (i.e. TNM) of cancer and if cancer present, then the CDS tool suggest which medication to take and suggests a treatment plan based on latest cancer research and latest treatment guidelines.  The CDS tool offers an explanation for its decision based on knowledge resources from the literature and patient data.  

Note: In a developing country and under-developed country, the above holds true, except there is a lack of access to oncologist. Thus, this CDS tool has even more relevance in these countries, as a general physician or a nurse practician is making cancer treatment decisions based on limited knowledge of treatments and drugs.

***Scenario 2***: A patient has been diagnosed with cancer and receiving chemotherapy.  The insurance company obtains the multimodal data from the patient, in return for additional reimbursements for their therapy and treatment.  The insurance company runs this data into the CDS tool to determine the possibility of future cancer recurrence and accordingly adjust the premiums of the patients.  Thus, not all cancer patients are paying a high insurance premium, but the premium is according to their specific condition.

***Scenario 3***: A patient who has been declared cancer-free, asks the medical professional their chances of cancer relapse or cancer spread. The professional can use the CDS tool to obtain generative images that would show if any chances of cancer relapse or spread. This is then communicated to the patient. 

***Scenario 4***: A patient has been cancer-free, but needs to do follow-up visits every 9 months with their medical professional.  The lab test, imaging data from these follow-up visits is fed into the CDS tool to determine the future trajectory of cancer.  Also, the patient has a wellness app, so they can weekly monitor their symptoms, if any.  Based on the entire medical history and these weekly data points, the CDS tool will inform the patient if they should talk to their doctor about these symptoms. Thus, the intervention is at the right-time, versus waiting for the next 6-to-9-month visit.

## Features In

[M] denotes minimum viable product features.

### Section 1: Data Input Features

1.1	[M] A medical professional in the Provider setting should be able to manually upload patient data from past, current and follow-up visits. Specifically, MRI/CT/X-rays imaging exams, genetic tests PDF reports, lab tests reports, doctor’s notes, radiologist PDF reports, pathology PDF reports, drugs, and demographics. 

1.2	In the Provider setting, the CDS tool should be integrated with the EMR (Electronic Medical Record), so that all the patient data is automatically imported into the CDS tool. If the automatic upload of data fails, then provide an option to manually upload the patient data.

1.3	The CDS tool should be integrated with the radiologist workflow in the PACS (Picture Archiving Communication Systems), LIS (Lab Information System), RIS (Radiology Imaging Suite) and the oncologist workflow in the EMR and OIS (Oncology Information System).

### Section 2: Data Handling Features

2.1	[M] Patient data should stay with the Provider control, either on cloud or on-premise. 

2.2	[M] Only AI model feedback data from the free-field text box, survey, thumbs up/down will be sent back to the company once a day.

### Section 3: Error Reporting Features

3.1 [M] In response to possible Context errors and Fail state errors, user shall have the ability to provide input via free-field text box, survey, thumbs up/down.

### Section 4: Model Output Features

4.1 [M] For each patient, the user will have the option to enable or disable whether they want the CDS tool to produce an output for the patient. 

4.2 [M] For each patient, the user will have the option whether they want to save the entire model output or part of the model output or not save at all.

4.3[M] Once a diagnostic image is acquired, an automated radiology report will be produced, which will be available for review only to the radiologist

4.4[M] Once a diagnostic image is acquired, if cancer is present, then automatic segmentation of the existing cancerous region will be shown for the review of radiologist

4.5 [M] For current patient visit, for a patient with confirmed cancer diagnosis, a textual output of whether person has continued presence of cancer will be shown.

4.6 For current patient visit, if a person has not yet been diagnosed with cancer, based on the symptoms presented a textual indication of whether person has cancer will be shown for review by the medical professional. 

4.7 For current patient visit, a patient who has been confirmed diagnosed with cancer, an image will be produced that shows the future trajectory of cancer

4.8 For current patient visit, a patient who has been declared cancer-free by medical professional, the likelihood of cancer re-appearing will be shown for review by medical professional. 

4.9 For current patient visit, a patient who has been confirmed diagnosed with cancer, an automated treatment plan with which drugs to be prescribed will be produced for review by medical professional.

4.10 For a patient, who has been diagnosed with cancer and who has been declared to be free of cancer, the CDS tool will produce a risk score for their relapse of cancer, and patient will have the option to automatically transfer this information and model to insurance companies. 

### Section 5: Explainability Features

5.1 [M] The model will produce three outputs for each of the above use cases and each will have the categorical label – high AI confidence, medium AI confidence, low AI confidence. Except, for the case where there is a decision on whether person has cancer or not – a single output will be produced with the above mentioned categorial labels.

5.2 [M] For each output, their will be an explanation of which data points in the patient history were used to arrive at the conclusion, for the medical professionals.  

5.3 [M] There will be a general description of how the model works. Also, their will be an example video of end-to-end how the model input and output works.

5.4 There will be a counterfactual way of explaining the output of the model. This involves explaining why the AI made a particular decision by showing how the decision would have changed if one or more input features had been different.

5.5 For each model output, the appropriate knowledge resources and guidelines will be mentioned as the source of reference on which the model output is based.

### Section 6: Wellness App Patient-Facing Features

6.1	[M] Patient Login credentials provided based on prescription by Physician and two-factor authentication to protect patient data.

6.2	[M] Patient provides consent to allow their entire medical history into the app, so recommendations are tailored to their medical condition at that time.

6.3	[M] Symptom tracking for Patient: The app should allow patients to track their cancer symptoms on a weekly basis. This could include symptoms such as pain, fatigue, nausea, vomiting, and diarrhea.

6.4	[M] Trend analysis: The app should be able to analyze the patient's symptom data over time to identify any concerning trends and show last 6 months window symptoms to patient, whereas, for the Physician the entire history of symptom tracking should be shown. For example, if the patient's pain is increasing or their fatigue is worsening.

6.5	[M] Physician notification: If the app detects a concerning trend, it should automatically notify the physician who prescribed the app. 

6.6	[M] Patient notification: The app should also notify the patient if it detects a concerning trend. This will encourage the patient to contact their physician as soon as possible.

6.7	Education: The app should provide patients with educational resources about cancer and its symptoms. This could include information about different types of cancer, how to manage symptoms, and how to stay healthy.

6.8	Support: The app should provide patients with a way to connect with other cancer patients and survivors. This could be done through a forum, chat room, or social media group.

6.9	Personalization: The app should be personalized to the individual patient's needs. This could include allowing the patient to customize the symptom tracking features, the educational resources, and the support options. 

6.10	Reminders:  If the patient has not logged on the app for two consecutive weeks, then send the text message or email reminder, based on patient preference. Similarly, if the Physician has not acknowledged the alert, then send them another alert 3 days later, for a total of maximum five alerts.

### Features Out

7.1	For the under-developed countries, with the nurse practician as the user persona, only one model output will be provided with minimal to no explanation, as they don’t have the expertise to understand it. 

7.2	No patient facing wellness app and insurance companies’ software features for the developing and under-developed countries.

7.3	Model output confidence accuracy in terms of percentage will not be shown.

7.4	Example-based justification will not be provided as it exposes the patient confidential training dataset used.  Even though it is an anonymized patient it would cause confusion for the user as after reviewing the suggested training dataset, they could conclude that their case at hand is quite dissimilar and hence will quickly mistrust the AI model. 

## Features Prioritization

![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/7efb89d4-0134-4594-b784-11a2027d4796)

## AI Model Metrics

•	The AI model shall have a Explainability score of at least 80% as measured by the user satisfaction expert review surveys. The System Causality scale of 1 to 10 will be used.  The LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (Shapely Additive Explanations) methods of measuring Explainability will also be explore. 

•	The AI model shall have a trustworthiness score of at least 90% as measured by auditing by expert reviewers’ survey.

•	The AI model shall have a fairness score of at least 90% as measured by auditing by expert reviewers’ survey.  The methods of Disparate Impact, Equalized odds, and Counterfactual fairness will also be explored.  Disparate impact measures whether AI models affects disproportionately certain groups of people. Equalized Odds measures whether AI product has the same accuracy for all groups of people. Counterfactual fairness measures whether AI product would make the same decision for two people with similar characteristics, except for one protected characteristic. 

•	The AI model shall have a recall accuracy of at least 80%

## Technical Considerations

•	Unsupervised algorithms will be utilized. 

•	This use case requires generative AI techniques.

•	Initially, preference will be given to a simpler AI model with the trade-off of lower accuracy compared to a complex model.  This is to build trust with the users with better explanations of the output of the simpler AI model. 

## Go-To-Market (5-year horizon)

Customers will be developed in the following phases:

***Phase A***: Select Large Integrated Delivery Network (IDN) Providers in the US.  The model will be validated in each of these sites individually before full paid roll-out. Third party partnerships in-place for CDS tool integration. 

***Phase B***: Select Providers in Canada and Europe, in order of priority, UK, Germany, Italy, and France. The model will be validated in each of these sites individually before full paid roll-out.

***Phase C***: Once the model matures, full paid roll-out in US and Europe customers. 

***Phase D***: Wellness app for select patients to validate model. Select insurance companies in the US for model validation.

***Phase E***: Once the model matures, full paid roll-out for insurance companies in US. Free full roll-out for patients facing wellness app.

***Phase F***: Select large providers in five developing countries. Once the model matures, full paid roll-out in five developing countries.

## Pricing

US and Europe Providers pay $25,000 base fee per year + $200 per patient on which CDS tool used.

Insurance companies pay $50,000 base fee per year + $100 per patient on which CDS tool used.

Patient facing wellness app will be free of cost as the information collected from this app will help insurance companies on an ongoing basis determine insurance premium. So, indirectly this app is being funded by insurance companies.

## Approaches: Tasks

Based on the Go-To-Market Phases, the development features and model development will be prioritized.  

### Action plan for objective# 1: “Integrate the tool into the workflow of at least 35% of Providers in the US.”

  Key Result #1.1: Desired data is procured for at least 500 diverse US patients and data pre-processing is complete.
  
  Key Result #1.2: The steps of feature engineering, model selection, model training and evaluation are complete.
  
  Key Result #1.3: Model development steps are re-performed for a 10,000-patient dataset.  Begin to develop customer relationships with Providers demoing the capability of the model from the 500 patient’s dataset.
  
  Key Result #1.4: The Minimum Viable Product [M] Features outlined above are complete.
  
  Key Result #1.5: Deploy model for evaluation at five Provider sites in the US. Begin the develop the Return-on-investment business case alongside the customer which would include understanding their willingness to pay. Develop almost all remaining features mentioned above.
  
  Key Result #1.6: Iterate on model improvements and feature enhancements based on customer feedback.
  
  Key Result #1.7: FDA (Food and Drug Administration) approval of the CDS tool. Model deployed at 5% of paying customer sites in US.
  
  Key Result #1.8: Model deployed at 10% of paying customer sites in US.
  
  Key Result #1.9: Model deployed at 20% of paying customer sites in US.
  
  Key Result #1.10: Model deployed at 35% of paying customer sites in US. At all sites it is used on at least 50% of the patients monthly.
  
  Key Result #1.11: Continuous model improvement in-house and subsequent revised deployment at customer sites.

### Action plan for objective# 2: “Integrate the tool into the workflow of at least 30% of Providers of Canada, UK, Germany, Italy, and France.”

  Key Result #2.1: Desired data is procured for at least 500 diverse patients from these countries and data pre-processing is complete.
  
  Key Result #2.2: Augment the US model with these new data points and perform the steps of feature engineering, model selection, model training and evaluation.
  
  Key Result #2.3: Use the US CDS tool features as a starting point and deploy at five sites in these European countries for model evaluation.
  
  Key Result #2.4: Iterate on model improvements and feature enhancements based on customer feedback.
  
  Key Result #2.5: EU CE-mark and TUV approval of the CDS tool. Model deployed at 5% of paying customer sites in US.
  
  Key Result #2.6: Model deployed at 10% of paying customer sites.
  
  Key Result #2.7: Model deployed at 20% of paying customer sites.
  
  Key Result #2.8: Model deployed at 35% of paying customer sites. At all sites it is used on at least 50% of the patients monthly.
  
  Key Result #2.9: Continuous model improvement in-house and subsequent revised deployment at customer sites.

### Action plan for objective# 3: “Adoption of a cancer wellness app by at least 60% of patients”

Key Result #3.1: Create the MVP features for Android and iOS mobile app and desktop.

Key Result #3.2: Test the UX of app on 100 patients and 20 Physicians.  Make changes to app based on feedback.

Key Result #3.3: Finalize the app and make it commercially available.

Key Result #3.4: Complete the rest of the features, test the app on similar sample size and make it commercially available.

Key Result #3.5: app maintenance and continuous improvement based on patient and physician feedback.

### Action plan for objective# 4: “Adoption of the CDS tool by at least 25% of US insurance companies”

Key Result #4.1: Develop Model Features for insurance companies.

 Key Result #4.2: Test features alongside three insurance companies.
 
Key Result #4.3: Tool adopted by 5% of insurance companies

Key Result #4.4: Tool adopted by 10% of insurance companies.  Continuous improvement to the model.

Key Result #4.5: Tool adopted by 25% of insurance companies. Continuous improvement to the model.

### Action plan for objective# 5: “Integrate the tool into the workflow of 10% of Providers in five developing countries and under-developed countries”

Key Result #5.1: Desired data is procured for at least 500 diverse patients from these countries and data pre-processing is complete.

Key Result #5.2: Augment the US and Europe model with these new data points and perform the steps of feature engineering, model selection, model training and evaluation.

Key Result #5.3: Use the US and Europe CDS tool features as a starting point and deploy at five sites for model evaluation.

Key Result #5.4: Iterate on model improvements and feature enhancements based on customer feedback.

Key Result #5.5: Local regulatory agency approval of the CDS tool. Model deployed at 2% of paying customer sites in US.

Key Result #5.6: Model deployed at 10% of paying customer sites.

## Design & Commercialization Roadmap
![image](https://github.com/vis6060/MedSegDiff_LiTS/assets/75966773/87c3a96e-60f6-4fb9-bd1f-3c8be46efd8d)

