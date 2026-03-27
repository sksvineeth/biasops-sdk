import sys
import click

@click.group()
def cli():
    """BiasOps — policy-as-code enforcement for AI governance."""
    pass

@cli.command()
@click.option("--model",      required=True)
@click.option("--X-test",     "x_test_path", required=True)
@click.option("--y-test",     "y_test_path", required=True)
@click.option("--policies",   multiple=True, required=True)
@click.option("--protected",  multiple=True, required=True)
@click.option("--domain",     default=None)
@click.option("--output-dir", default=".")
@click.option("--warn-only",  is_flag=True, default=False)
@click.option("--policy-dir", default=None)
@click.option("--y-col",      default=None)
def check(model, x_test_path, y_test_path, policies, protected,
          domain, output_dir, warn_only, policy_dir, y_col):
    """Evaluate a model against BiasOps compliance policies. Exits 0=pass, 1=block."""
    import joblib
    import pandas as pd
    from .biasops import eval as biasops_eval, BiasOpsBlockError

    def _load(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
        click.echo(f"Unsupported format: {path}", err=True)
        sys.exit(2)

    try:
        fitted_model = joblib.load(model)
        X_test = _load(x_test_path)
        y_raw  = _load(y_test_path)
        y_test = y_raw.iloc[:,0] if isinstance(y_raw, pd.DataFrame) else y_raw
        y_pred = fitted_model.predict(X_test)
        biasops_eval(model=fitted_model, X_test=X_test, y_test=y_test, y_pred=y_pred,
                     policies=list(policies), protected_cols=list(protected),
                     domain=domain, output_dir=output_dir,
                     warn_only=warn_only, policy_dir=policy_dir)
        sys.exit(0)
    except BiasOpsBlockError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"BiasOps error: {e}", err=True)
        sys.exit(2)

def main():
    cli()

if __name__ == "__main__":
    main()
