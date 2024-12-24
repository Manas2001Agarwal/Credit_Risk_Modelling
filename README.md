End-END machine learning classification project

Developed a Credit Risk ClassificaAon Model: Built a machine learning model to classify bank customers into 4 risk levels (P0, P1, P2, P3) based on their credit history, leveraging a real-world ICICI Bank dataset.

Data Preprocessing & Feature Engineering: Utilized Pandas for data cleaning, performed Sequential VIF-based feature selection, reducing dimensionality from 64 to 39 features, and mitigated multicollinearity to prevent overfitting.

Modeling & Hyperparameter Tuning: Applied CatBoost to train the model, optimized performance through hyperparameter tuning, and tracked experiments using MLflow, achieving 78.39% accuracy on unseen data.

Deployment & Automation: Developed a Streamlit front-end for model interaction, containerized the application with Docker, and automated build and deployment using GitHub Actions and Docker Hub for streamlined CI/CD.

Link to Streamlit App: https://manas2001agarwal-ml-project-app-zx6nu8.streamlit.app/                     (This is to run my ML Application directly)
Link to Docker File: https://hub.docker.com/repository/docker/manas2512/credit-risk-modeling/general     (This is to pull docker image of my application from DockerHub)
Link To DagsHub Repo : https://dagshub.com/Manas2001Agarwal/Credit_Risk_Modelling/experiments            (This is to check out and see comparison of ML Experiments, I tracked using                                                                                                             ML-Flow)
