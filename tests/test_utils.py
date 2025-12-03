from bioguider.utils.utils import convert_to_serializable
from bioguider.utils.constants import ProjectMetadata

def test_convert_to_serializable():
    project_metadata = ProjectMetadata(
        url="https://github.com/OpenBMB/RepoAgent",
        project_type="application",
        primary_language="python",
    )
    serialized_project_metadata = convert_to_serializable({
        "overview": {"touched": True, "evaluation": project_metadata},
    })
    assert serialized_project_metadata is not None
    assert serialized_project_metadata["overview"]["touched"] == True
    assert serialized_project_metadata["overview"]["evaluation"]["url"] == "https://github.com/OpenBMB/RepoAgent"
    assert serialized_project_metadata["overview"]["evaluation"]["project_type"] == "application"
    assert serialized_project_metadata["overview"]["evaluation"]["primary_language"] == "python"