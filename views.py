import os

from enum import Enum

class MenuOptions(Enum):
    LOAD_DATA = 1
    CHANGE_PARAMS = 2
    TRAIN_MODEL = 3
    SHOW_RESULTS = 4
    EXIT = 5

class View():
    def clear_screen(self) -> None:
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

    def show_header(self) -> None:
        print("============================================================")
        print("                 ML Reconocimiento de Dígitos               ")
        print("============================================================")

    def show_footer(self) -> None:
        print("============================================================")
        print("               Fin de la aplicación. ¡Gracias!              ")
        print("                  (C) Fernando A. Gómez F.                  ")
        print("============================================================")

    def show_message(self, message: str) -> None:
        print(f"\n*** {message} ***\n")

    def display_menu(self) -> MenuOptions:
        menu: MenuOptions = None
        while menu is None:
            print("*** Menú de opciones: elija una opción:")
            print("\t1. Cargar set de datos de dígitos")
            print("\t2. Cambiar parámetros del modelo")
            print("\t3. Entrenar modelo")
            print("\t4. Mostrar resultados")
            print("\t5. Salir")
            print("Ingrese una opción: ")
            try:
                cmd = input(":> ")
                menu = MenuOptions(int(cmd))
            except ValueError:
                print("\n** Opción inválida, intente de nuevo. **\n")
                menu = None
        return menu 
    
    def _capture_value_i(self, value_name:str, tip:str, min_val:int, max_val:int) -> int:
        captured:bool = False 
        val:int = 0
        while not captured:
            try:
                print(f"\t{value_name} ({tip}):")
                cmd = input(":> ")
                val = int(cmd)
                if not (min_val <= val <= max_val):
                    raise ValueError("Valor fuera de rango.")
                captured = True
            except ValueError:
                print(f"*** Valor inválido para {value_name}, intente de nuevo. ***")
        
        return val
    
    def _capture_value_f(self, value_name:str, tip:str, min_val:float, max_val:float) -> float:
        captured:bool = False 
        val:float = 0.0
        while not captured:
            try:
                print(f"\t{value_name} ({tip}):")
                cmd = input(":> ")
                val = float(cmd)
                if not (min_val <= val <= max_val):
                    raise ValueError("Valor fuera de rango.")
                captured = True
            except ValueError:
                print(f"*** Valor inválido para {value_name}, intente de nuevo. ***")
        
        return val
    
    def capture_parameters(self) -> dict:
        params = {}

        params["seed"] = self._capture_value_i("Semilla generador aleatorio", "entero > 0", 1, 2**32 - 1)
        params["test_size"] = self._capture_value_f("Tamaño del set de prueba", "valor entre 0 y 1", 0.0, 1.0)

        return params