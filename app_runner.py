import os
import sys
import streamlit.web.cli as stcli
import argparse


def main():
    # Configuración de argparse para capturar el parámetro --mode
    parser = argparse.ArgumentParser(description="Modo de ejecución de la app")
    parser.add_argument(
        "--mode",
        choices=["local", "mongo"],
        default="local",  # Valor por defecto será MongoDB
        help="Especifica si usar MongoDB o el cargador local (local)",
    )

    args = parser.parse_args()

    # Crear archivo de configuración temporal para que `app.py` lo lea
    with open("config_runtime.py", "w") as f:
        f.write(f"USE_LOCAL = {args.mode == 'local'}\n")

    # Lanza Streamlit
    sys.argv = ["streamlit", "run", "web_app/app.py"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
