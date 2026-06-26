# Hiring Intelligence Dashboard

The project reimagines resume analysis by shifting focus from vague AI suggestions to data-driven insights based on real hiring outcomes. Rather than trying to predict job offers directly, we analyze which resume traits most strongly correlate with success—using actual recruitment data, SHAP interpretability, and behavioral scoring. Our tool empowers job seekers not just with feedback, but with explanations and comparisons, allowing them to simulate changes, view trait benchmarks against successful applicants, and identify potential bias or inequality in the hiring process. We aim to move beyond generative feedback and toward transparent, evidence-backed recommendations. This provides applicants with quantifiable and context-aware recommendations. 

## Project Description:

By using **Natural Language Processing (NLP), machine learning models, and statistical analysis**, the project aims to create an actionable and ethical tool that helps job seekers **optimize their resumes** while also understanding **what actually matters in hiring decisions** and addressing **potential biases in the hiring process.**

## 1. Data Selection and Cleaning
All raw data is stored in `.csv` format under the `dataset/raw_data/` directory.

### Raw Datasets

#### `job_data.csv`
- **Source**: [Kaggle - Job Descriptions](https://www.kaggle.com/datasets/marcocavaco/scraped-job-descriptions)
- **Contents**: Job descriptions and categories  
- **Usage**: Generate BERT embeddings to compare with user resume embeddings.

#### `jobss.csv`
- **Source**: [Kaggle - Predicting Job Titles](https://www.kaggle.com/datasets/thedevastator/predicting-job-titles-from-resumes/data)
- **Contents**: Job titles, categories, and required skills  
- **Usage**: Identify high-frequency skills missing from user resumes.

#### `Recruitment_data.csv`
- **Source**: [Kaggle - Hiring Decisions](https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data)
- **Contents**: Applicant features and hiring outcomes  
- **Usage**: Train a LightGBM model and apply SHAP to interpret hiring decisions.

#### `Resume_data.csv`
- **Source**: [Kaggle - Resume Dataset](https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset)
- **Contents**: Structured resume information (skills, schools, timestamps)  
- **Usage**: Visualize applicant demographics and qualifications.

#### `Resume.csv`
- **Contents**: Plain-text resumes and their associated categories  
- **Usage**: Extract top keywords and generate embeddings for peer comparison.


---

### Processed Datasets (`dataset/processed_data/`)
- `Cleaned_job_data.csv`: Cleaned version of `job_data.csv`
- `Jobss_cleaned.csv`: Cleaned version of `jobss.csv`
- `Plain_resume.csv`: Cleaned version of `Resume.csv`
- `Cleaned_recruitment_data.csv`: Cleaned version of `Recruitment_data.csv`
- `Encoded_cleaned_recruitment_data.csv`: Categoricals encoded for modeling
- `Cleaned_updated_resume_data.csv`: Cleaned version of `Resume_data.csv`

---

### Keyword Files (`/keywords/`)
- `Cleaned_jobdata_keyword_freq.csv`: Top 50 keywords by job category (from `extract_jd_keywords.ipynb`)
- `Industry_top_skill_keywords.csv`: Top 50 skills per job category (from `extract_keywordOfSkills_in_jobss_cleaned.ipynb`)
- `Plain_resume_keyword_freq.csv`: Top 50 resume keywords per category (from `extract_keyword_plain_resume.ipynb`)

---

### Gender Subsets (`/subsets_by_gender/`)
- Subsets used for SHAP analysis in Part 3  
- Explained in `Subset creation and analysis Ivan.ipynb`

## 2. Statistical Analysis and Data Visualization

All the files that are addressed in this part can be found under the file called ‘notebooks’:

**File 1 - Directory: final\_visualization/encoded\_cleaned\_recrutment.ipynb:** 

**To get the following images, run each cell of the ipynb file sequentially.**

#### Scatter Plot: SkillScore vs. InterviewScore by Hiring Decision
- **Purpose**: Visualize how skill and interview scores affect hiring decisions.
- **Insight**: Hired candidates tend to cluster in the upper-right quadrant, showing high values in both scores.
- **Explanation**: Each point represents a candidate, with their SkillScore on the x-axis and InterviewScore on the y-axis. Circles represent candidates who were hired, while X’s represent those who were not. This plot reveals clusters and thresholds associated with success.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXd8AXzcDDEopBFXc5i0ptISzFxu26dTKn0xJS5nJwsO-enavgi3DMadYVQd9dK2zJFu5oSzvdL5YTejB3hwYnOYQzTWqM9jZ7yoYqDT6xuqtRYzfYbfuCNcfn37k1NUo7lbkuj3?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Key Takeaways:_

  - Hired candidates cluster in the upper-right quadrant, where both skill and interview scores are high.

  - Very few candidates with low SkillScore are hired, regardless of their Interview Score.

  - There are a few edge cases where high-skill candidates were not hired, likely due to other factors (e.g. personality, fit, recruitment channel).

* _Implication for Analysis:_ This plot suggests that SkillScore and InterviewScore are strong indicators of hiring likelihood, especially when both are above \~70.

#### Boxplot of PersonalityScore by EducationLevel
- **Purpose**: Analyze variance in personality scores across education backgrounds.
- **Insight**: PhD and Master’s holders show higher and more consistent scores than Bachelor and High School levels.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcLAk9r2jEF02h_7IpESWbn2PQolqXFeHztVNfXp3M6AlxtcMfEMHZz32jk39wlCHvDNJgfBsMIbmDID0HB_6JBgdJ_LWDK7vgZbJxbGaxNFF7euhF3SI_9-TOf-gJPln9NWWsb?key=AydKl4-5KABXmLdy7Lo3PixK)

- **Explanation**:_ This box plot shows the distribution of PersonalityScore across EducationLevel groups (e.g. High School, Bachelor, Master, PhD). It includes medians, quartiles, and outliers, highlighting consistency and variability.

* _Key Takeaways:_

  - PhD and Master’s degree holders tend to have higher and more consistent personality scores.

  - Bachelor and HighSchool groups show wider variability and lower medians.

  - The presence of outliers indicates occasional strong candidates across all levels.

- _Implication for Analysis:_ This plot supports the hypothesis that higher education may correlate with stronger interpersonal traits, though variability within lower education levels suggests that strong personalities can emerge from all backgrounds.

#### Bar Chart: Hiring Rate by Recruitment Strategy and Education Level
- **Purpose**: Evaluate how recruitment method impacts hiring success across different education levels.
- **Insight**: Referrals consistently result in the highest hiring rates.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcnxBprKMHWIOtHuxtucft-KUHHnhG_8Fi_G63w8gPdZTKDn5MIQMZcdwPHpY82BVtR5qeafDKxjv0T1HHj_ae0byxNYHGC4cJzfGDgJyBG814-b9BWbUt53nxE02X7bvj4C9L0HQ?key=AydKl4-5KABXmLdy7Lo3PixK)

- **Explanation**:_ This grouped bar chart shows the percentage of candidates hired for each RecruitmentStrategy (Referral, Online, Agency), with separate bars for each EducationLevel. It reveals how strategy and education intersect to influence hiring success.

* _Key Takeaways:_

  - Referral-based candidates consistently have the highest hiring rates across all education levels.

  - Online applications are moderately effective, while agency hired tend to have the lowest success rates.

  - The gap between strategies is most pronounced among Bachelor and Master degree holders.

- _Implication for Analysis:_ Referral programs yield higher success rates should be prioritized for sourcing candidates, especially for mid-to-senior roles.

#### Parallel Coordinates Plot: Hired vs. Not Hired Profiles
- **Purpose**: Compare multivariate candidate profiles across InterviewScore, SkillScore, PersonalityScore, and ExperienceYears.
- **Insight**: Hired candidates have stronger metrics overall, especially in InterviewScore and SkillScore.
- **Explanation**:_ Each line represents a candidate’s normalized values across four metrics: InterviewScore, SkillScore, PersonalityScore, and ExperienceYears. Lines are color-coded by hiring outcome. This allows you to visually trace what makes a “typical” hired candidate different from a successful one.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeZ1oNREhXFDN2-f5PPVGYsxqd6o6IWQiTPN08XSL3zZ6PdiwPfgsGXXSHk5AGe1tTXeSsZtgv5fR4RgIY2NUorvOt2603JhxXdpqYIBluw6MmOOiZ3RnlLx64_gpZ7nIV7ObpthQ?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Key Takeaways:_

  - Hired candidates consistently score higher across all metrics, especially SkillScore and InterviewScore.

  - Differences in ExperienceYears and PersonalityScore are present but less stark. 

  - The visualization clearly separates groups, suggesting that a weighted combination of these metrics could effectively predict hiring outcomes.

* _Implication for Analysis:_ This confirms that successful candidates follow a multi-metric profile. A scoring algorithm or ML model can leverage these patterns for decision making or recommendations.

**File 2 - Directory: mid-term\_visualization/visualization.py:**

**To get the visualization, run this .py file, and the result will be under ‘datasets/visuals\_images’**

#### Resume Keyword Frequency Analysis

As part of our data exploration, we conducted a keyword frequency analysis on the raw resume text. We used a custom keyword list to identify how often programming languages and machine learning (ML) concepts appeared across all candidates’ resumes. This analysis helped us understand the technical depth and focus areas in the applicant pool.

1. **Top Programming Languages (Keyword-Based):**

_What it shows:_ This bar chart displays the most frequently mentioned programming languages based on keyword presence in resume text.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcIiCrZwuM27BjE0aHb1-oD8EedYJ1ePE5DyQMl12EFW7NpV_sjkj-y-yDmf7ZD6JJ8GckZkdyuY4x7BN2LIwjjFkA5-ut9Bn4SD7TYma-fKEHCnCGnN9inUROTHJFKFCZWrvE?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Method:_

  - We searched each resume (after cleaning) for mentions of common programming languages like “Python”, “Java”, “C++”, “SQL”, etc.

  - Mentions were case-intensive and counted even if they appeared as part of larger phrases.

* _Key Takeaways:_

  - Python dominates the candidate pool, followed by SQL, Java and JavaScript.

  - Languages like C++ and R appear less frequently, suggesting most candidates come from data-centric or full-stack backgrounds.

2. **Top Machine Learning Skills (Keyword-Based):**

_What it shows:_ This chart highlights the most common ML techniques and concepts mentioned across resumes. 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeHRpbtZf5ZbmUXSLX_HhxyMvFtNPG5XvDC99v-MJbD-jMbHgv1yMEwVfvkZ5pn5_1Za6ML7Bg7xasfTdF0xWl0OFdNOw2yXyFePRqkDWBJza32k8kcTzCEomaOvOYAZprSVUB14w?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Method:_

  - We compiled a targeted list of keywords such as “regression”, “SVM”, “random forest”, “topic modeling”, “neural network”, etc.

  - The code scanned each resume’s text and tallied keyword occurrences to find the most prevalent techniques.

- _Key Takeaways:_

  - Regression, clustering, and random forest are the most cited techniques.

  - Advanced topics like deep learning, topic modeling (LDA/NMF), and neural networks also appear frequently, reflecting a technically skilled applicant base.

3. **Why It Matters:**

This analysis was important for two reasons:

1. It helped validate the presence of high-value skills across resumes, which we later tied to hiring outcomes and recruitment strategies.

2. It informed the feature engineering process for our final model by surfacing keywords we could use to quantify domain expertise.

**File 3 - Directory: mid-term\_visualization/visualization\_jobss.py:**

**Job Market Landscape Analysis:**

This section summarizes our exploratory data analysis (EDA) of a cleaned job listings data. This visualizations help contextualize demand patterns in job roles, industries, experience requirements, and the specific sets requested by employers.

1. **Top 10 Most Common Job Roles:**

- _Purpose:_ Identify which job titles are most frequently offered in the dataset.

- _Insights:_ 

  - Shows which roles dominate the current hiring landscape.

  - Useful for aligning resume optimization with high-frequency opportunities.

- _Example:_ Roles like Software Developer, Data Analyst, and Project Manager appear at the top, indicating consistent market demand.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcwfKeow3Fj6Uac0s7DDRUv21dSnp56McR1DS8vY703P6VZc1UMAt_T7gEpDPbxMi1P9kpI75gJNCnHAPJ9TU0rihIgtKokrH_o_zeQg4I6hZMUvhuvgYSdXbte4UjdVJmXA-SNBA?key=AydKl4-5KABXmLdy7Lo3PixK)

2. **Distribution of Minimum Experience Requirements:**

- _Purpose:_ Visualize how much prior experience is typically required for job listings.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdOUWWQTJFq2MYZq6qkkWVc2m9wilnkHQyMtb4tr9ncBOOcgYtk5ZchczuJf4H2EycdNgEcbwneyIpsM7qFdYA0_kiDgcfPtk5C5k41Xt0DVF-PBwliFwVSeMNBow5Zqds9CzsP?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Insights:_

  - Most roles require between 0-5 years of experience.

  - The histogram is skewed right, confirming that many roles are entry-to-mid level positions.

- _Use:_ Helps filter which resumes should be benchmarked against which roles during evaluation.

3. **Top 10 Industries with Most Job Openings:**

- _Purpose:_ Explore which industries are hiring most actively.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdmN9d-ZvnTgtYc257EglunYTUX1V78ONJYjxMgTNUCMVMOGvYRrshPPCiSu6gMl1BvQzLBnO6mhSxAevOopkp8ncs9Z9xqJjp6Vy4ffysNtWCXSRgblAq6seUSCx5UvkFTHqxbHQ?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Insights:_

  - Tech and IT services dominate, but finance, healthcare, and education also appear prominently.

  - This helps position a candidate’s experience toward industry-aligned roles.

4. **Top 15 Most Required Skills:**

- _Purpose_: Understand the technical and soft skills in highest demand.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfX7W7W1EYshmWiNvmxuS4Wp7h5A91r0muIR3jZ2KYWx_R1ZyflRNMsqQ21aMjXmKcC39WqRKmJQndPVrhAQLn19XcGFzW0IZ9z25AX4aF6XuXnwfN6oOvfvGVxkxy_bTbHJcvX?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Insights:_

  - Skills like communication, python, SQL and problem-solving top the list.

  - Reflects both hard and soft skills crucial to employers.

- _Use:_ Useful for weighting skills in resume evaluation or for guiding candidates to fill in knowledge gaps.

5. **Experience Requirements by Role Category:**

- _Purpose:_ Compare experience thresholds across job categories:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcM1OEwbh9qZRkCXzOfMRUGGZnefDYH4lCkvGrpsIx1JlffgrlB3R_YeydF9QBgRbsnyAJgI-_UvnfK6WiWk2Hi1juTQTDzQrZuhAiozfL9rpWjLHD7vB_zSKc2BaNzjXE5T6Fq?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Insights:_

  - Roles in Engineering and IT demand higher experience.

  - Support or entry-level roles cluster around 0-2 years experience.

- _Use:_ Important for segmenting evaluation logic based on target role complexity.

6. **Top 10 Functional Areas:**

- _Purpose:_ Highlight the most common job functions (e.g., Engineering, Sales, Operations)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeDZCrU84_Z30fE0UjLNKQ90neDdoRu3FqHOE1r-SGhCtjtI14Sf2e5UelJJSg4ZRPKDBoQK5McyWuKG1DVKoIB9RH_mT-wetRDQbNAJHu6CpyLsO9CuYVB-UpJdSv4OlSFJQy5?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Insights:_ 

  - Emphasizes demand for tech-centric roles (Engineering, Data) as well as business units like Sales and HR.

  - Use for candidates planning to pivot to high-opportunity domains.

7. **Number of Skills vs. Experience Level:**

- _Purpose:_ Analyze how the number of required skills varies by years of experience.

****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe5j-4n7brFMS41P95b7-_3-_NEzlyhonM7P9yi_dbofMzpUfswizH7_QS3KK4LGeCtaNIbL1Jg5RwgUN01WhwVFNV-AsogIdGgrgASkzJYERl8bV2zKrOjK55FWdMhgJChJ5gU0w?key=AydKl4-5KABXmLdy7Lo3PixK)****

- _Insights:_

  - Entry-level jobs (0-2 years) require fewer skills (\~5-6 on average).

  - Senior roles (8+ years) list up to 10-12 required skills.

  - Confirms that complexity scales with experience.

8. **Top 3 Skills Required for Top 10 Job Roles:**

- _Purpose:_ Break down skill demand by specific job titles.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXc9rKAL-k4-MoAyLRPbAFW4ZIEgugLdanJBg9Zoe3NFVh5cblx0v2-KJFuEpMeL5YG5xv2v_ei38IrReXzpkfhNE62mHegZNqMZGwzdjO6mI-_m4-cIEUBilH2B1QbbH4y2VUn-pg?key=AydKl4-5KABXmLdy7Lo3PixK)

- _Insights:_

  - For example, Python, SQL and Data Visualization dominate Data Analyst roles.

  - Each job title has a distinct top-3 skills “signature”.

- _Use:_ Essential for resume-tailoring: match candidate skills to the exact traits employers expect per role.

**Summary**

_This analysis helped us:_

- Contextualize our resume evaluation dataset against real market demand.

- Guide feature selection (e.g., which skills to encode).

- Understand how experience, skills and roles interconnect, improving control-group logic for evaluating hiring fairness.

**File 4, 5, 6 - Directory: ****keyword\_extract/extract\_jd\_keywords.ipynb &**

******keyword\_extract/extract\_keyword\_plain\_resume.ipynb& ****keyword\_extract/extract\_keywordOfSkills\_in\_jobss\_cleaned.ipynb**

In these three files, I:

**A.** From cleaned\_job\_data.csv, extracted the top 50 keywords and their counts for each category’s job descriptions, and saved the result to

‘datasets/processed\_data/keywords/cleaned\_jobdata\_keyword\_freq.csv’

**B.** From plain\_resume.csv, extracted the top 50 keywords and their counts for each category’s resumes, and saved the result to

‘datasets/processed\_data/keywords/plain\_resume\_keyword\_freq.csv’

**C.** From jobss\_cleaned.csv, extracted the top 50 required skills and their counts for each category, and saved the result to

‘datasets/processed\_data/keywords/plain\_resume\_keyword\_freq.csv’

All the keywords will be crucial references when we are analyzing user’s resume in 

‘notebooks/user\_resume\_analyze.ipynb’. Detailed Explanation can be found in part 4.

**—-How to run:**

To get all the keywords csv, run every cell sequentially. 

**File 7- Directory: noteboooks/mid-term-visualization/visualize\_resume\_keyword.ipynb**

I applied word‐cloud visualizations to the file datasets/processed\_data/keywords/plain\_resume\_keyword\_freq.csv, 

generated a word cloud of the top keywords for each category, and saved all the images to 

datasets/visuals\_images/wordcloud\_resume\_keyword. The result looks like this:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfXd1NCtwjsiDjjM6owMcQJx7ctevNfl_wLMAFrLiuAYD5flkuGZdgz4P7IkdDcpoO9HqB5KvEW9JWY_5mWRr6_AuXvs-ekA6yGFGCNdfZJh2dKcveoyoNBeaGUjKIRgQ4qnGj4?key=AydKl4-5KABXmLdy7Lo3PixK)

**—-How to run:**

To get all the word-cloud visualization images, run every cell sequentially.  

********

## 3. Data Modeling and SHAP Analysis

All the following files are under the file of notebooks:

‘SHAP.py’: this is the groundwork of the following two files, so no need to run this file

In ‘**SHAP\_revised\_by\_zetian\_0428.ipynb’**:

Reads the encoded recruitment data from “datasets/processed\_data/encoded\_cleaned\_recruitment\_data.csv”, which contains columns: (ExperienceYears, InterviewScore, SkillScore, PersonalityScore, Gender\_encoded, EducationLevel\_encoded, RecruitmentStrategy\_encoded) and target (HiringDecision).

Gender\_encoded: 1 = male, 0 = female\
RecruitmentStrategy\_encoded: 2 = referral, 1 = online, 0 = agency\
EducationLevel\_encoded: 0 = High School, 1 = Bachelor, 2 = Master, 3 = PhD\
\
Trains a LightGBM classifier Prints a classification report (precision, recall, F1)\
Plots feature importance by gain via lgb.plot\_importance

Uses SHAP to compute and plot a summary of how each feature influences predictions:****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe1dzqLD66c9NmVBKzR0umrvTHECO5U2XgAYWH38oo_n8S9qzoXncEqR0HprZY25ppS-7WybaV8ShEd8UY74zZVQQQQZYmC_e9BHz1fln-48yZseuhDiWrt_JlsH1P4j93TXhh3jw?key=AydKl4-5KABXmLdy7Lo3PixK)****

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdzW3QD4uvX9IgHKqUq--fswOOTK3pwUfGiDzYLgADoD1O8S6GZsi1iPl15k9cUhut3Pqhy6KGmNXhaEClWdfILWCRWNsVY4H-DreCEWa0ynH9TC80HH9cqE8W_W-pQMv5dvB8jnw?key=AydKl4-5KABXmLdy7Lo3PixK)

From both analyses, it’s clear that **RecruitmentStrategy** is the most important factor (i.e. whether an applicant applies online, through an agency, or via referral). If someone is referred (i.e. has a company referral letter), their success rate increases dramatically. **Personality**, **skill**, and **interview** scores are far less important than referrals. Interestingly, **education level** is not as influential as **experience years** or any of the factors above. Finally, **gender** has almost no impact on hiring outcomes, reflecting that the modern labor market does not exhibit severe bias.

Building on this, the next notebook will explore subsets of the data, for example:

1. “For candidates with referrals, which factors are most important?”

2. “For male versus female applicants, which factors differ most in their impact on outcomes?”

**“Subset creation and analysis Ivan.ipynb”** : 

**All the visualizations generated in this file can also be found via this path:**

**datasets/visuals\_images/subset analysis visualization**

**They are in the “.ipynb” file as well.**

This is the file where we create subsets of the dataset and perform SHAP analysis on each of the subsets.

File relative path: noteboooks/subset creation and analysis Ivan.ipynb

We realize that in “recruirtment\_data.csv”, there are many interesting columns as listed above. As we want to offer suggestions to certain demographics of people, we decided to create subsets based on genders and see what are the most important factors for male or female applicants and will boost the possibility to get them hired.

Here’s the detailed explanation of the codes:

We first save the cleaned version of the encoded recruitment data. And then we set the path to save the new subsets and create the subsets. As we discovered from the “SHAP\_revised\_by\_zetian\_0428.ipynb” that females and males have different influences on the final model output, for the first set of subsets we decided to separate male applicants from female applicants. We iterate through each unique gender in the cleaned data to get the features and save the files. Then we train and apply the LGBM in the same manner as we did to the whole dataset but this time on the subsets.

For females, we first read the female subset file and make sure that we don’t include the gender-related column to avoid extraction. Then we split the train and test set before performing the LGBM model.

Then we plot the graphs:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXf1tyFspw6sRN4NI0FoNEBpRXmGJww8mOafjiWkh9Y8sK9BBwDa5lBPkziHY0Im5cM9eEG3Ref2fyS8VNhv0y_GAKIT6ff0u0tfGKq1jSuXW4yplsJqLSIVWKn21nWNwdyHULCc?key=AydKl4-5KABXmLdy7Lo3PixK)

We can see from the graph that recruitment strategy is the most important factor for females, way more important than the other factors. And education level is the least important factor. 

Then we calculate the SHAP values and plotted it:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXd7jzQSI8QOCdvU0636M6lbYGvedegZnD0r1RUJM1j42NWUpD-UutSrdpXa5_SgPoRb51IZIrutzXEz8FOrLkT2H2tl78reH9r9IrmMM5BDcqxR9Nt5rv9BoIMY-naDNQvROGr4SQ?key=AydKl4-5KABXmLdy7Lo3PixK)

For all the features, it is generally that the higher the feature values, the more positive impact they have on the model output, which does make sense because the features all represent something positive. It would benefit the companies to hire people with a higher positive score.

But the three recruitment strategies are “online”, “referral”, and “agency”, which do not necessarily have a “positive” or “negative” inference.

As the recruitment method is also the most important deciding factor to the hiring outcome according to the LGBM analysis, we think it is meaningful to further analyze the subsets of each recruitment strategy under each subset of genders. So we first applied this to the female subset.

There are three different recruitment strategies: referral(encoded with 2), online(encoded with 1), and agency(encoded with 0). We basically repeated the same process as before, just three times for three recruitment strategies this time(we are using the female subset as the base). 

Here is the output of the female who do referral:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdiA9Py9c7ZkJiqwa9qGsU6kwwf1Vor67n16begm3jSY-X1haIJzQcEAQMnw3aNxUOEYgcedhCIN3ofcBwvD8vze8CHxY_jQy0ydSolR9IsCdTElVDqZZEm_HpTa6CfrZdvj5h0Zw?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfmyVCtU4pcs4zEbtrPSVekq3BOMV5mSaGioZWX4mvpFb2Zidg0u2ApKlofY3c732FsMhBBqDXABScRfOVYW57b0YdEZxyQJv0zTnQNgwVuylwmcZRTqMv8QqDmHjHrggVqECcO?key=AydKl4-5KABXmLdy7Lo3PixK)

For this demographic, the rank of feature importance is: SkillScore, ExperienceYears, PersonalityScore, InterviewScore, and EducationLevel. 

For these features, it is generally that the impact is positively correlated with the feature value. However, sometimes a high interview score might lead to negative influences on the final outcome.

Here is the output of the female who do online:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdzv6ICW63jiUcBnn_5y9MIXESrtHrI2sSzNOtw6bjkUudf4ZwbXoJwMgcFdZUbuMDsgjH1gJEOsLjxjmbMBsXucj_O-WytHkllC3Kl7R3NI8sgAPotbBgPKxN-JbDD6wsZIXBl?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfENgs_d_OIiG8OloCGDlUJYtZ03E39TgfjqINxCyqQRCgbsFAYuEKkqy9d8NDCoNXTUOln1liqn2K5f_ziw44lrfCLTkbnOVVHUGdFjME-r7x8z2xF03qMy4SCdLyce9g1UknB?key=AydKl4-5KABXmLdy7Lo3PixK)

For this demographic, the rank of feature importance is: SkillScore, InterviewScore, PersonalityScore, EducationLevel, and ExperienceYears.

For the females who do online, high interview score, high skillscore, and longer experience years might harm the hiring outcome. But a good education level would almost guarantee a positive influence on the outcome.

We think it also makes sense because the employers might not trust the performance of the applicants during interviews as much since it is held online. However, education level is something that pre-existed and cannot easily fake.

Here is the output of the female who do agency:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXduVgYNMt4XzG4xffPwe_X-RwgvCvuny1KuvVo_lrwKoSwd1GVVE3U0XnRQVM6qo7-aKaHR3eWshIDS1kSqRwECvbB14OYeD3H62vOz3C0g0Xapwod1WpZ_rbwEqicogbr-Q86KGA?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfrTCdCs27-tDqGI7N5cfhZARi2cEBfmfAiLSd5apDGdIFoL8XyRMooYUU4srUEKvhMMUUQnyck1IDFoTyLkZXkNpgsojb6hhz1GHlcR6rjs7o_o7Ymv2K3-FBpQmGE-DCSg2sJXQ?key=AydKl4-5KABXmLdy7Lo3PixK)

For this demographic, the rank of feature importance is: PersonalityScore, ExperienceYears, SkillScore, EducationLevel, InterviewScore, and ExperienceYears.

For the females who do online, it is true that the higher the scores are, the more positive the influences they have on the outcome are. It matches the common sense.

Then we conducted the same set of operations on the male subset:

Here is the output for the male group in general:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfoGYR9Bl-TSAGqsNwZSPqu8c5InogGnZR0hO7OIA1N7N0wv--WWEJ0DlKIhPq0X8GHm1obAfHS_lLfkny2Q4q6JqScXl0fuRmoKuiGIG_DCYbVEaz221WKMjXpta64aJKhAXSsnw?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdIlN44C9tySzz3mrU5wVmKE6j8bSDhiPpTDhi2Di7qJ3hYfphZ5zfacHJuoTRduKtKJNdwhq7g6iELvCrhH8Xt4g_YEP_Jtf_HancnJXrZew-P-1BQ9Dv-d5d7Jf2kNCXbAPZppQ?key=AydKl4-5KABXmLdy7Lo3PixK)

The two graphs match really well. In both of the graphs, the recruitment strategy is the most important factor. A high value of the encoded strategy (red) pushes predictions strongly positive, and a low value (blue) pushes strongly negative. The rank here is recruitment strategy, SkillScore, InterviewScore, ExperienceYears, and EducationLevel. PersonalityScore is the least impactful for males because its SHAP values cluster more tightly around zero, indicating it barely moves the prediction.

Then within the male demographic, we further subgrouped it according to the recruitment strategies. Here are our discoveries:

For males who do referrals:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcr_St59WZoOSl-w7Ri9aZLomxaoCzYVZ9bVOTZZLXseV1Ge55esYc_1kMiqUdqJMqZav7u5wNB9yiHNMtqAUPwHygf-7zr8sJxkvVdbD28OQWuVpAAC4L3Kr4ayd1wTof2hO6jyg?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXccBAiQ1Kn5MH-fXt6AH7GniklqN8dXawIyt5jbQpICuOjYU9BH5e5VW2NOx5dKiUFIP3bTWN_3t9SZxzo78nY_Kjlb0qh7E1ew_Q7x4VL9YgIJaEEvla636Ok8fmcEEeZ3m26r?key=AydKl4-5KABXmLdy7Lo3PixK)

According to LGBM, the importance rank is SkillScore, ExperienceYears, PersonalityScore,  InterviewScore, and EducationLevel\_encoded. And the SHAP summary chart also looks conventional. InterviewScore and PersonalityScore both have some noises, but in general the SHAP values and the feature values are positively correlated.

For males who do onlines:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfp8QRaFsEhsUIemyZhjnByRqdUIDh3kmwGhQa46jZHLhPyJI6Swb3ikKLk9eKVP_bmSOoNtlSzXiJLAk_6G4JKTevLkFBURqKUZ0XKg_CtrhU6G5yDMpQo--EmnzxdKkZjL3WZcw?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeKNLJS-jre4NnGYQdLyayBr-Ez2NYICqMO9VOJ0flL671XOEdQO8wOF0LUB8rVVl7QDUr1EfPjBoiCVAFUdVeEx0N9rJYN0l0XlwgPRR1WCJ5skbWVVIJglubq81mcF1dOGGcnkQ?key=AydKl4-5KABXmLdy7Lo3PixK)

According to LGBM’s gain for the “Male – online” group, the features rank as PersonalityScore ,InterviewScore, SkillScore, EducationLevel\_encoded, and ExperienceYears  

In the SHAP summary plot ExperienceYears has the largest mean SHAP value since high years push positive more and  low years push negative more. As PersonalityScore’s dots are close to zero, it is the weakest mover. So among male applicants who choose to do interviews online, experience years are way more important than personalities.

For males who do agencies:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdMt2n802Ig3CcdJIV6ZW9y0fRUsyziORZ2d90U0ViYiyEPR0B6ax1Rhz4NIjl5KE2VgBPTqHYv_v6A6YX-3CTzCgJIp5fNMoP-8oQAMdNyIs6EKK_VXShfhgKj8tfoulZvJHzX?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeAqmkOr9yMeAkCkAe0RCANl9y41L-V2qlgA4YRe5xvC9WbMO26xmLQTtoc3i9EO_uuJkPjTuLr6zDEeeb1q-1cjZRT9flw9eAOZPiCJBrEiXimqeKUC6nFKozrVl8vlOPsUuINLQ?key=AydKl4-5KABXmLdy7Lo3PixK)

From the LBGM plot, the rank is InterviewScore, EducationLevel, SkillScore, ExperienceYears, and PersonalityScore. Although InterviewScore dominates the LGBM, ExperienceYears actually moves predictions the most on average. PersonalityScore barely budges the model either way (its dots hug the vertical zero line). So it is safe to say that for male applicants who do agency, the experience year is the key factor that influences the hiring outcomes.

Then, to make the visualization clearer, we plotted the male outcome and the female outcome together side by side.

Here is the visualization:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeuBKDZCri-Bec34IaX8AjH4-ehlwQF6ivNy9ZG8XmcOTYvT4zD3EJ-vf8QWzASNAgQUZBl3ykXfQmz3RAmTqnUaiKUksIi8DToiVpb8p_zxNxY3G2YRmiqETpaQq2K_2coS0O0Fw?key=AydKl4-5KABXmLdy7Lo3PixK)

In this way, it is clearer that it is very close, but male applicants have an advantage over female ones.

Finally, we look at the recruitment data as a whole in general and subgroup them based on the recruitment strategies in the same manner as before.

For those who use referrals:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe8e6Us1GW-WiTGr3Zd070Nxd8gkyx6twWJ2pahH_mhyBpb4vJXlbOdGZ1kkM0_Zx-o6UZzohNmbKwoCS452mikVGOoNa6i_Dg_tNYSFlqtqXDI17LxkmtunfTbiJL4qkWunehA?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcUWsHGHrbesXqpO0mj3A3Hh71QGn8ONe17MWyM-BZJlV_qWchbwtDwhvnPFTB2ykbYYTd40p8Ur40Z53z2JOL_XVGyNoPzu78MXLEmQVZqNDFPbiaFggiAhWh4P0AIWb31A6xVFg?key=AydKl4-5KABXmLdy7Lo3PixK)

Both the LBGM and the SHAP summary show that SkillScore is by far the strongest predictor for referrals, with PersonalityScore and InterviewScore next in the gain ranking.

However, SHAP shows the true impact order as SkillScore, InterviewScore, EducationLevel\_encoded, PersonalityScore, and ExperienceYears. 

In contrast, gender doesn’t seem to influence the hiring decision when every applicant is under referral.

For those who do online:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdGulylcQlSCrmcS30A5mDD0lkOf6qI3dpBWxNtO4Vig0L_PjbBxeVx8rPJ-opQ8-Oxv7G5FRvAZHvrXfkjlGCq4wzr9vXWSm2RiIS8CmKaD8ZZCHvAtwNtyBkmCo_hAcSjKZmMMg?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe26VLSs6-TaaNMplswNb7-25q2XilmeOziLJ_b1JlWIMIs3n_h3zTsWLyYOZkWX2ViN6UkQ7g1wFqgTsJjskRqsS91dZn20vtm7DeBKiaAdaYpO82NuH4RO-rlwRbWVzy_ZGL5hw?key=AydKl4-5KABXmLdy7Lo3PixK)

Both the LGBM and the SHAP summary for the online subset agree that SkillScore and InterviewScore dominate the tree splits, followed by PersonalityScore, EducationLevel\_encoded, ExperienceYears, and finally Gender\_encoded. 

However, SHAP reveals the true average contribution order is InterviewScore, PersonalityScore, EducationLevel\_encoded, SkillScore, ExperienceYears, and Gender\_encoded, with higher feature values (red) pushing predictions up and lower (blue) pushing them down. A higher interview score does not guarantee that the hiring outcome would be good, which is interesting.

Same as before, gender doesn’t influence much here.

For those who do agency:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXf8UXpHizOaDlgz_o2Q58IRgCQj2SgfHR38oljyNWHAqqxo-3OKE3HGq2QWVsDLiqOHrL1NDbXGbQbhkqsaiLLpmfQoCV5Noh0dSAq0Uw-xpyDoX8YAFULmTOpP_gssSnKyvlDRNw?key=AydKl4-5KABXmLdy7Lo3PixK)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeA4UZ7_Yf3LfB4uPWrczoHzKzlgPa_A5uup2BQ18zznlVUmsS9NXjO3yEXpxyt_NP1CNEyilhHUKIxryZE3njcb66Im4LkOXHw-Oj3Ph3S8XoL3kfE1Sr3aS-ir6tOPmfUbNK3OQ?key=AydKl4-5KABXmLdy7Lo3PixK)

Both the LGBM and the SHAP summary for the agency subset agree that PersonalityScore and InterviewScore dominate the tree splits, followed by EducationLevel\_encoded, SkillScore, ExperienceYears, and Gender\_encoded. 

However, SHAP reveals the true average contribution order is InterviewScore, PersonalityScore, SkillScore, EducationLevel\_encoded, ExperienceYears, and finally Gender\_encoded, with higher feature values (red) pushing the prediction up and lower (blue) pushing it down. 

Again, gender doesn’t have a big influence on the result.

## 4. Application - user resume analysis

This part  contains a single notebook:

‘notebooks/user\_resume\_analyze.ipynb’. 

It automates a two-fold evaluation of your resume—semantic alignment and keyword coverage—against industry data. 

**1. Category Matching**

• Prompts you to enter your desired role (e.g. “Software Engineer”, “Data Scientist”, “UX Designer”). 

• Uses a BERT-based encoder to find the closest match among three system datasets: job descriptions, peer resumes, and skill requirements.

**2. Resume Ingestion** 

• Reads your resume.txt from datasets/user\_resume/. 

• Tokenizes and embeds your full text for downstream analysis. 

**3. Semantic Similarity Scoring** 

• Aggregates all job descriptions in the matched category and computes a cosine similarity score between this combined text embedding and your resume embedding. 

• Repeats the process for the collection of peer resumes to gauge how your profile aligns with industry standards. 

**4. Keyword Gap Analysis** 

• Loads pre-computed top keywords (with frequencies) from: 

datasets/processed\_data/keywords/cleaned\_jobdata\_keyword\_freq.csv 

datasets/processed\_data/keywords/plain\_resume\_keyword\_freq.csv 

datasets/processed\_data/keywords/industry\_top\_skill\_keywords.csv

 • Parses each list and identifies missing high-frequency terms in your resume by applying substring checks and an embedding-based similarity threshold. 

• Highlights “key keywords” that are absent across multiple categories as priority areas to strengthen. 

**5. Report Generation** 

• Prints concise similarity scores for both job descriptions and peer resumes. 

• Lists missing keywords per category alongside their frequencies.

 • Offers targeted recommendations on which skills or terms to consider adding. 

**How to run:** 

1\. Place your resume.txt in datasets/user\_resume/ and update its path in cell 

2\. Run all cells except the last one. 

3\. Execute the final cell and enter your category when prompted to see the full analysis.a
