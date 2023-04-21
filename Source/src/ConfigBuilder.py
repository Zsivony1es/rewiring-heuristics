from typing import Optional


class ConfigBuilder():

    class AppConfig():
        def __init__(self, model_type: str, generate_plots: bool) -> None:
            self.model_type = model_type
            self.generate_plots = generate_plots

    def __init__(self, cmd_input: Optional[str] = None, prop_path: Optional[str] = None) -> None:
        if prop_path != None:
            self._parse_properties_file(prop_path)
        if cmd_input != None:
            self._parse_cli_properties(cmd_input)

    def _parse_cli_properties(self, input: str) -> None:
        parm_list: list[str] = str.split(input, sep=" ")
        # Parse arguments while they exist         
        while parm_list:
            match tuple(parm_list):
                case ("--modelType", model_type, *rest):
                    self._model_type = model_type
                    parm_list = rest
                    continue
                case ("--generatePlots", *rest):
                    self._generate_plots = True
                    parm_list = rest
                    continue
                case _:
                    raise RecursionError("Couldn't parse command line options")
        return
              
    def _parse_properties_file(self, path: str):
        prop = dict(line.strip().split('=', maxsplit=1) for line in open(path))
        self._model_type = prop['model.type']
        self._generate_plots = True if prop['generate.plots'] == 'True' else False
         
    def build(self) -> AppConfig:
        return ConfigBuilder.AppConfig(
             model_type=self._model_type,
             generate_plots=self._generate_plots
        )
        
             
            

    

  
 