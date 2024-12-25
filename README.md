End-END machine learning classification project

Link to Streamlit App: https://manas2001agarwal-ml-project-app-zx6nu8.streamlit.app/                     (This is to run my ML Application directly)

Link to Docker File: https://hub.docker.com/repository/docker/manas2512/credit-risk-modeling/general     (This is to pull docker image of my application from DockerHub)

Link To DagsHub Repo : https://dagshub.com/Manas2001Agarwal/Credit_Risk_Modelling/experiments            (This is to check out and see comparison of ML Experiments, I tracked using                                                                                                             ML-Flow)


**Confusion Matrix + Classification Metrics on Test Data**
<img width="1011" alt="image" src="https://github.com/user-attachments/assets/85664e5c-3e73-4f6d-8b15-c364c9b27e38" />


Developed a Credit Risk Classification Model: I designed and implemented a machine learning model aimed at classifying bank customers into four distinct risk levels (P0, P1, P2, and P3) based on their credit history. This model was built using a real-world dataset provided by ICICI Bank, offering a practical foundation for understanding customer creditworthiness.

Data Preprocessing & Feature Engineering: To ensure the model's effectiveness, I used Pandas for comprehensive data cleaning and preparation. I then applied a Sequential VIF-based feature selection approach, which reduced the dataset's dimensionality from 64 features to 39. This reduction helped to mitigate multicollinearity issues, ensuring that the model would not suffer from overfitting due to highly correlated features. This step significantly enhanced the model’s predictive performance by focusing on the most relevant features.

Modeling & Hyperparameter Tuning: For model training, I leveraged CatBoost, a powerful gradient boosting algorithm. I fine-tuned the model’s hyperparameters to improve its performance and reduce bias or variance. Throughout this process, I utilized MLflow to track experiments, log parameters, and monitor performance. After rigorous tuning, the model achieved 78.39% accuracy on unseen data, demonstrating strong generalization capability.

Deployment & Automation: Once the model was trained, I developed a user-friendly Streamlit front-end interface, allowing stakeholders to interact with the model and input new data for predictions. To ensure smooth and scalable deployment, I containerized the application using Docker. This provided a consistent and portable environment for the model. Additionally, I automated the build and deployment pipelines using GitHub Actions and Docker Hub, enabling streamlined CI/CD processes and ensuring that the application could be quickly updated and deployed without manual intervention.
