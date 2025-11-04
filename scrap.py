from run_experiment_ALL_animals import main as run_exp
from run_evaluation_with_all_empirical_networks import main as run_eval

run_exp()

# 0,Rat4,
# 103,Orangutan2,
# 169,RedKangaroo3,
# 188,FruitBat5,
# 206,Chimpanzee,

for animal_id in [206, 0, 188, 169, 103]:
    print(f"\n\n=== Running evaluation for animal ID: {animal_id} ===\n")
    run_eval(experiment_name=f"72_clean_10_000_animal_{animal_id}")

w