from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,val_data = obj.initiate_data_ingestion()
    
    data_transformation  = DataTransformation()
    train_arr,val_arr,_ = data_transformation.initiate_data_transformation(train_data,val_data)
    
    model_train = ModelTrainer()
    print(model_train.initiate_model_training(train_arr,val_arr))
    