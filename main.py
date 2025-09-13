
from controllers import Controller
from views import View 
from models import Model

class Application():
    _controller:Controller = None

    def __init__(self):
        model = Model()
        model.load()
        view = View()
        self._controller = Controller(model, view)

    def run(self) -> None:        
        self._controller.run()

if __name__ == "__main__":    
    app = Application()
    app.run()
