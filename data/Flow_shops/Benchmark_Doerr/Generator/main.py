import pathlib

import generator

if __name__ == "__main__":
    exec_file = "C:/Program Files/IBM/ILOG/CPLEX_Studio221/cpoptimizer/bin/x64_win64/cpoptimizer.exe"
    dirname = pathlib.Path(__file__).parent.parent.resolve()
    target = str(dirname) + '/Instances/'


    # gen = generator.Generator(target, '', solver='OR-Tools')
    gen = generator.Generator(target, exec_file, solver='Docplex.cp')
    # gen.generate_instance('default_problem_08', layout_dict={'seed': 42}, orderbook_dict={'seed' : 42}, uncertainty_dict={'ratiominmean_setup': 0.8, 'ratiominmean_processing': 0.8})
    
    read_path = str(dirname) + '/Instances/default_problem_08.json'
    gen.generate_uncertainty('default_problem_08', read_path, uncertainty_dict={'ratiominmean_setup': 0.8, 'ratiominmean_processing': 0.8})
    gen.generate_uncertainty('default_problem_07', read_path, uncertainty_dict={'ratiominmean_setup': 0.7, 'ratiominmean_processing': 0.7})
    gen.generate_uncertainty('default_problem_06', read_path, uncertainty_dict={'ratiominmean_setup': 0.6, 'ratiominmean_processing': 0.6})
    gen.generate_uncertainty('default_problem_05', read_path, uncertainty_dict={'ratiominmean_setup': 0.5, 'ratiominmean_processing': 0.5})

