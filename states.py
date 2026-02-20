from FeatureCloud.app.engine.app import AppState, app_state
from logic import fl_algorithm

# Please do not implement ANY other state! Keep all logic in the called fl_algorithm function.
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        # /mnt/input and /mnt/output are the default input and output folders
        # that featurecloud uses
        fl_algorithm(
            fed_learning_class_instance=self,
            inputfolder="/mnt/input",
            outputfolder="/mnt/output")
        return 'terminal'
        # This means we are done. If the coordinator transitions into the
        # 'terminal' state, the whole computation will be shut down.
