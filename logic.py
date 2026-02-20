from typing import Optional

from helper.protocolfedlearningclass import ProtocolFedLearning

def fl_algorithm(fed_learning_class_instance: ProtocolFedLearning,
         inputfolder: Optional[str] = None,
         outputfolder: Optional[str] = None):
    """
    Main function to run the federated learning protocol.
    Can also use helper classes and is can be called in the AppState classes
    or in a simulation script.
    """
    raise NotImplementedError("Please implement the federated learning logic. Use the fed_learning_class_instance to call federated learning related methods.")
