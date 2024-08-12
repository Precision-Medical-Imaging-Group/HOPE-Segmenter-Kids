"""MLCube handler file"""
import os 
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'  # see issue #152
import typer
import os

app = typer.Typer()


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    output_path: str = typer.Option(..., "--output_path"),
):
    challenge = 'ped'
    
    if (challenge in ["ped", "PED", "BraTS2023-PED"]):
        from ped_runner import setup_model_weights, batch_processor
    
    setup_model_weights()
    batch_processor(data_path, output_path)

    return output_path



@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
