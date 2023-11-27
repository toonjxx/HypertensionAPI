import torch 
import numpy as np 
from typing import Dict, Any

class Inference:
    def __init__(self, package: dict) -> None:
        self.model = package['model']
        self.classmap = {
            0: 'low risk of hypertension',
            1: 'high risk of hypertension'
        }
    
    def _preprocess_request(self, request : Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess request

        Args:
            package (dict): The package used by FastAPI to get the scaler 
            request (Dict[str, Any]): The request we will be getting 

        Returns:
            Dict[str, Any]: Preprocessed request
        """
        print(request)
        data = request['ppg']

        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).to('cuda')
    

    def predict(self, request: dict) -> dict:
        preprocessed_request = self._preprocess_request(request)
        model = self.model
        model.eval()
        model.to('cuda')
        predictions = model(preprocessed_request)
        predicted_class = torch.sigmoid(predictions).round()
        return {
            'prediction' : self.classmap[predicted_class]
        }