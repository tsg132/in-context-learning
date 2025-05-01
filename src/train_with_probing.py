import os
from random import randint
import torch.nn as nn

from torch.utils.data import DataLoader
import uuid

from quinine import QuinineArgumentParser
from probe_dataset import get_probe_dataset
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def freeze_layers_except_last(model):
    """Freezes all layers except the last layer(s) for linear probing."""
    for name, param in model.named_parameters():
        if 'head' not in name and 'final' not in name and 'output' not in name:  # Common names for final layers
            param.requires_grad = False
    
    # Log number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, probe):

    probe_head, probe_opt, probe_loss_fn, probe_loader = probe
    # If linear probing is enabled, freeze all layers except the last
    if args.training.get('linear_probe', False):
        print("Enabling linear probing - freezing all layers except last...")
        freeze_layers_except_last(model)
        
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],  # Only optimize unfrozen parameters
        lr=args.training.learning_rate
    )
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)

    task_kwargs = dict(args.training.task_kwargs)

    if args.training.task == "rff_regression_fixed":

        from tasks import RFFRegressionFixed

        fixed_pool = RFFRegressionFixed.generate_pool_dict(n_dims, args.training.num_tasks, args.training.rff_dim)

        task_kwargs["pool_dict"] = fixed_pool

    task_sampler = get_task_sampler(

        args.training.task,
        n_dims,

        bsize,

        num_tasks = args.training.num_tasks,

        **task_kwargs
    )

    # task_sampler = get_task_sampler(
    #     args.training.task,
    #     n_dims,
    #     bsize,
    #     num_tasks=args.training.num_tasks,
    #     **args.training.task_kwargs,
    # )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        if probe_head is not None and i % args.probe.log_every_steps == 0:
            xb, yb = next(iter(probe_loader))

            xb, yb = xb.cuda(), yb.cuda()
            with torch.no_grad():
                feats = model.extract_features(xb)

            logits = probe_head(feats)

            p_loss = probe_loss_fn(logits, yb)

            probe_opt.zero_grad()

            p_loss.backward()

            probe_opt.step()

            acc = (logits.argmax(-1) == yb).float().mean().item()

            wandb.log({
                "probe/loss": p_loss.item(),
                "probe/acc": acc,
            }, step=i)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    
    # Load pretrained weights if specified
    if args.model.get('pretrained_path', None):
        print(f"Loading pretrained weights from {args.model.pretrained_path}")
        state_dict = torch.load(args.model.pretrained_path)
        # Handle both cases where state_dict is direct or nested in 'model_state_dict'
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        
    model.cuda()
    model.train()

    if getattr(args, "probe", None):

        for p in model.parameters():

            p.requires_grad = False

        probe_head = nn.Linear(model.n_dims, args.probe.num_classes).cuda()

        probe_head.train()

        probe_opt = torch.optim.Adam(probe_head.parameters(), lr=args.probe.lr)

        # probe_loss_fn = nn.CrossEntropyLoss()

        probe_loss_fn = nn.MSELoss()

        probe_dataset = get_probe_dataset(args.probe.dataset)

        probe_loader = DataLoader(probe_dataset, batch_size=args.probe.batch_size, shuffle=True)

    else:

        probe_head = probe_opt = probe_loss_fn = probe_loader = None

    train(model, args, (probe_head, probe_opt, probe_loss_fn, probe_loader))

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
