"""
Reproduce the experiments

"""

import json
from gpt4or import *
from configure import api_keys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
parser.add_argument(
    "--path", type=str, default="./datasets/lectures_in_lp_modeling/problem_7"
)
parser.add_argument("--maxtry", type=int, default=5)
parser.add_argument("--aug", type=int, default=5)
parser.add_argument("--keyid", type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    prefix = "description"
    starting_path = os.getcwd()
    print("Using API Key %d: %s" % (args.keyid, api_keys[args.keyid]), flush=True)

    test_modes = [
        # MODE_STANDARD_PROMPT,
        MODE_COT_ONLY,
        MODE_COT_DEBUG,
        MODE_COT_DEBUG_TEST,
        MODE_COT_HUMAN,
    ]

    # test_modes = [MODE_COT_DEBUG]
    n_modes = len(test_modes)
    n_instances = 1
    all_status = {"Aug-%d" % i: [] for i in range(n_instances)}
    all_iters = {"Aug-%d" % i: [] for i in range(n_instances)}

    # Solve under different modes
    n_aug = 0

    for mode in test_modes:
        # os.chdir(starting_path)

        num_augment = args.aug
        if mode != MODE_COT_HUMAN:
            num_augment = 1

        info = [STATUS_SYNTAX_ERROR, -1]

        for n_aug in range(num_augment):
            os.chdir(starting_path)
            # Augmentation
            if n_aug == 0:
                std_file = "%s.txt" % prefix
            else:
                std_file = "%s-r%d.txt" % (prefix, n_aug)

            try:
                or_agent = GPT4OR(
                    model_name=args.model, solver="gurobi", api_key=api_keys[args.keyid]
                )
                info = or_agent.solve_problem(
                    problem_path=args.path,
                    problem_file=std_file,
                    bench_file="test-human.py",
                    max_attempt=args.maxtry,
                    solver="gurobi",
                    solve_mode=mode,
                    solve_params=None,
                )

                if info[0] == STATUS_PASSED:
                    break

            except Exception as e:
                info = [STATUS_SYNTAX_ERROR, -1]
                print(e, flush=True)

        # all_status["Aug-%d" % 0].append(info[0])
        # all_iters["Aug-%d" % 0].append(info[1])

        print(
            "Mode %d Iter %d Status %d Try %d" % (mode, info[1], info[0], n_aug),
            flush=True,
        )

    os.chdir(starting_path)

    # # Dump to json
    # with open(os.path.join(args.path, "status.json"), "w") as f:
    #     json.dump(all_status, f)
    #     f.close()
    # with open(os.path.join(args.path, "iter.json"), "w") as f:
    #     json.dump(all_iters, f)
    #     f.close()

    # Write to csv
    with open(os.path.join(args.path, "res.csv"), "w") as f:
        for key, val in all_status.items():
            ins_status = all_status[key]
            ins_iters = all_iters[key]
            for i in range(len(ins_status)):
                f.write("%d," % ins_iters[i])
                f.write("%d," % ins_status[i])
            f.write("\n")
