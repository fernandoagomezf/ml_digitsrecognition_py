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
            try:     
                choice = self._view.display_menu()
                self._view.clear_screen()
                match (choice):
                    case MenuOptions.LOAD_DATA:
                        self.load_data()
                    case MenuOptions.CHANGE_PARAMS:
                        self.change_params()
                    case MenuOptions.TRAIN_MODEL:
                        self.train()
                    case MenuOptions.EVALUATE:
                        self.evaluate()
                    case MenuOptions.EXIT:
                        self.exit()
            except Exception as e:
                self._view.show_message(f"Error: {str(e)}")
                self._view.pause()
                self._view.clear_screen()
            
    def load_data(self) -> None:
        self._view.clear_screen()
        self._model.load()
        self._view.show_message("Modelo cargado correctamente.")

    def change_params(self) -> None:
        self._view.clear_screen()
        params = self._view.capture_parameters()
        self._model.set_seed(params["seed"])
        self._model.set_test_size(params["test_size"])
        self._model.set_k_neighbors(params["k_neighbors"])
        self._view.show_message("Parámetros cambiados correctamente.")

    def train(self) -> None:
        self._view.clear_screen()
        self._view.show_message("Entrenando el modelo...")
        self._model.train()
        self._view.show_message("Modelo entrenado correctamente.")

    def evaluate(self) -> None:
        self._view.clear_screen()
        self._view.show_message("Evaluando el modelo...")
        self._model.evaluate()
        result = self._model.get_result()
        self._view.show_results(result)
        self._view.pause()

    def exit(self) -> None:
        self._running = False
        self._view.clear_screen()
        self._view.show_footer()
        