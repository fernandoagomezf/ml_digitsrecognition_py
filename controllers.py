from models import Model
from views import View
from views import MenuOptions

class Controller():
    _model:Model
    _view:View
    _running:bool 

    def __init__(self, model:Model=None, view:View=None):
        self._model = model if model is not None else Model()
        self._view = view if view is not None else View()
        self._running = False

    def run(self) -> None:
        self._running = True

        self._view.clear_screen()
        self._view.show_header()
        self._model.load()
        
        while self._running:            
            choice = self._view.display_menu()
            self._view.clear_screen()
            match (choice):
                case MenuOptions.LOAD_DATA:
                    self.load_data()
                case MenuOptions.CHANGE_PARAMS:
                    self.change_params()
                case MenuOptions.TRAIN_MODEL:
                    self.train()
                case MenuOptions.SHOW_RESULTS:
                    self.show_results()
                case MenuOptions.EXIT:
                    self.exit()
            

    def load_data(self) -> None:
        pass 

    def change_params(self) -> None:
        pass 

    def train(self) -> None:
        pass

    def show_results(self) -> None:
        pass 

    def exit(self) -> None:
        self._running = False
        self._view.clear_screen()
        self._view.show_footer()
        