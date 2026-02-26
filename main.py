from __future__ import annotations

import argparse

from core.param_space import parse_grid_spec, parse_params_json, parse_sweep_values
from core.utils import parse_csv_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Set Cover 单算法参数实验框架 V2")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen", help="生成数据集")
    p_gen.add_argument("--config", default="configs/dataset_profiles.yaml")
    p_gen.add_argument("--output-root", default=None)
    p_gen.add_argument("--dataset-id", default=None)
    p_gen.add_argument("--samples-per-class", type=int, default=None)
    p_gen.add_argument("--seed", type=int, default=None)

    p_run = sub.add_parser("run-single", help="运行单算法参数实验")
    p_run.add_argument("--config", default="configs/experiment_single.yaml")
    p_run.add_argument("--algorithm-id", default=None)
    p_run.add_argument("--algorithm-module", default=None)
    p_run.add_argument("--dataset-root", default=None)
    p_run.add_argument("--class-filter", default=None, help="逗号分隔类别过滤")
    p_run.add_argument("--mode", choices=["ofat", "grid"], default=None)
    p_run.add_argument("--repeats", type=int, default=None)
    p_run.add_argument("--seeds", default=None, help="逗号分隔，如 0,1,2,3,4")
    p_run.add_argument("--base-params", default=None,
                       help='JSON，如 {"alpha":1.0}')
    p_run.add_argument("--sweep-param", default=None)
    p_run.add_argument("--sweep-values", default=None,
                       help="逗号分隔，如 0.1,0.2,0.3")
    p_run.add_argument("--grid-spec", default=None,
                       help="如 alpha=0.1,0.2;beta=1,2")
    p_run.add_argument("--output-root", default=None)
    p_run.add_argument("--with-plots", dest="with_plots", action="store_true")
    p_run.add_argument("--no-plots", dest="with_plots", action="store_false")
    p_run.set_defaults(with_plots=None)

    p_plot = sub.add_parser("plot", help="基于结果 CSV 重绘图")
    p_plot.add_argument("--experiment-dir", required=True)

    p_multi = sub.add_parser("run-multi", help="一键运行多算法并自动横向对比")
    p_multi.add_argument("--config", default="configs/experiment_multi.yaml")
    p_multi.add_argument("--dataset-root", default=None)
    p_multi.add_argument("--class-filter", default=None, help="逗号分隔类别过滤")
    p_multi.add_argument("--algorithms", default=None,
                         help="逗号分隔算法 id 过滤，如 ga,hgasa")
    p_multi.add_argument("--repeats", type=int, default=None)
    p_multi.add_argument("--seeds", default=None, help="逗号分隔，如 0,1,2,3,4")
    p_multi.add_argument("--output-root", default=None)
    p_multi.add_argument("--with-single-plots",
                         dest="with_single_plots", action="store_true")
    p_multi.add_argument("--no-single-plots",
                         dest="with_single_plots", action="store_false")
    p_multi.set_defaults(with_single_plots=None)
    p_multi.add_argument("--with-compare-plots",
                         dest="with_compare_plots", action="store_true")
    p_multi.add_argument("--no-compare-plots",
                         dest="with_compare_plots", action="store_false")
    p_multi.set_defaults(with_compare_plots=None)

    p_compare = sub.add_parser("compare", help="对多个单算法实验目录做离线横向对比")
    p_compare.add_argument("--run-dirs", required=True, help="逗号分隔实验目录")
    p_compare.add_argument("--output-dir", default=None)
    p_compare.add_argument(
        "--with-plots", dest="with_plots", action="store_true")
    p_compare.add_argument(
        "--no-plots", dest="with_plots", action="store_false")
    p_compare.set_defaults(with_plots=None)

    return parser


def cmd_gen(args: argparse.Namespace) -> None:
    from core.generator import generate_dataset

    dataset_dir = generate_dataset(
        config_path=args.config,
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        samples_per_class_override=args.samples_per_class,
        seed_override=args.seed,
    )
    print(f"数据生成完成: {dataset_dir}")


def cmd_run_single(args: argparse.Namespace) -> None:
    from core.runner_single_algo import run_single_experiment

    overrides: dict[str, object] = {}

    if args.algorithm_id is not None:
        overrides["algorithm.id"] = args.algorithm_id
    if args.algorithm_module is not None:
        overrides["algorithm.module"] = args.algorithm_module
    if args.dataset_root is not None:
        overrides["dataset.root"] = args.dataset_root
    if args.class_filter is not None:
        overrides["dataset.class_filter"] = parse_csv_list(
            args.class_filter, cast=str)
    if args.mode is not None:
        overrides["experiment.mode"] = args.mode
    if args.repeats is not None:
        overrides["experiment.repeats"] = args.repeats
    if args.seeds is not None:
        overrides["experiment.seeds"] = parse_csv_list(args.seeds, cast=int)
    if args.output_root is not None:
        overrides["output.root"] = args.output_root
    if args.with_plots is not None:
        overrides["output.generate_plots"] = bool(args.with_plots)

    if args.base_params is not None:
        overrides["experiment.ofat.base_params"] = parse_params_json(
            args.base_params)
    if args.sweep_param is not None:
        overrides["experiment.ofat.sweep_param"] = args.sweep_param
    if args.sweep_values is not None:
        overrides["experiment.ofat.sweep_values"] = parse_sweep_values(
            args.sweep_values)
    if args.grid_spec is not None:
        overrides["experiment.grid.params"] = parse_grid_spec(args.grid_spec)

    run_dir = run_single_experiment(
        config_path=args.config, overrides=overrides)
    print(f"单算法实验完成: {run_dir}")


def cmd_plot(args: argparse.Namespace) -> None:
    from core.visualize import plot_from_experiment_dir

    paths = plot_from_experiment_dir(args.experiment_dir)
    print("绘图完成:")
    for path in paths:
        print(path)


def cmd_run_multi(args: argparse.Namespace) -> None:
    from core.runner_multi_algo import run_multi_experiment

    overrides: dict[str, object] = {}
    if args.dataset_root is not None:
        overrides["dataset.root"] = args.dataset_root
    if args.class_filter is not None:
        overrides["dataset.class_filter"] = parse_csv_list(
            args.class_filter, cast=str)
    if args.algorithms is not None:
        overrides["experiment.algorithm_filter"] = parse_csv_list(
            args.algorithms, cast=str)
    if args.repeats is not None:
        overrides["experiment.repeats"] = args.repeats
    if args.seeds is not None:
        overrides["experiment.seeds"] = parse_csv_list(args.seeds, cast=int)
    if args.output_root is not None:
        overrides["output.root"] = args.output_root
    if args.with_single_plots is not None:
        overrides["output.generate_single_plots"] = bool(
            args.with_single_plots)
    if args.with_compare_plots is not None:
        overrides["output.generate_compare_plots"] = bool(
            args.with_compare_plots)

    batch_dir = run_multi_experiment(
        config_path=args.config, overrides=overrides)
    print(f"多算法批量实验完成: {batch_dir}")


def cmd_compare(args: argparse.Namespace) -> None:
    from core.compare_multi_algo import compare_experiments

    run_dirs = parse_csv_list(args.run_dirs, cast=str)
    out_dir = compare_experiments(
        run_dirs=run_dirs,
        output_dir=args.output_dir,
        with_plots=True if args.with_plots is None else bool(args.with_plots),
    )
    print(f"多算法对比完成: {out_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "gen":
        cmd_gen(args)
    elif args.command == "run-single":
        cmd_run_single(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "run-multi":
        cmd_run_multi(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
