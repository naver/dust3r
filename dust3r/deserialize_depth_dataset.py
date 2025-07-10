import abc
import boto3
from urllib.parse import urlparse
import cv2
import numpy as np
import torch
from io import BytesIO
from zstandard import ZstdCompressor, ZstdDecompressor


class AbstractResourceManager(abc.ABC):
    @abc.abstractmethod
    def get(self, uri: str) -> bytes:
        raise NotImplementedError


class Boto3ResourceManager(AbstractResourceManager):
    def __init__(self):
        self.s3_client = boto3.client("s3")

    def get(self, s3_uri: str) -> bytes:
        parsed_uri = urlparse(s3_uri)
        if parsed_uri.scheme != "s3":
            raise ValueError(f"URI scheme must be s3, not {parsed_uri.scheme}")
        bucket = parsed_uri.netloc
        key = parsed_uri.path.lstrip("/")
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()


def unpack_bytes_np(compressed_bytes: bytes):
    return np.load(BytesIO(compressed_bytes), allow_pickle=True)

def zstd_decompress_bytes(compressed_bytes: bytes) -> bytes:
    return ZstdDecompressor().decompress(compressed_bytes)

def zstd_decompress_arr(compressed_bytes: bytes) -> np.ndarray:
    return unpack_bytes_np(zstd_decompress_bytes(compressed_bytes))

def deserialize_and_download_image(
    s3_uri: str, bit_depth: int, resource_manager: AbstractResourceManager, dtype: torch.dtype
) -> torch.Tensor:
    """Shared utility for DeserializedObjectView and DeserializedImage.

    Look at those class docstrings for more information.
    """
    image_bytes = resource_manager.get(s3_uri)
    if bit_depth == 8:
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    elif bit_depth > 8 and bit_depth <= 16:
        # note that torch half starts losing precision for bit_depth > 11; it becomes a choice for the user to
        # tradeoff loading speed vs precision. For bit_depth=12, the max error is 1px (out of 4096 slots).
        if dtype not in {torch.float, torch.half}:
            raise ValueError(f"dtype must be torch.float or torch.half if bit_depth > 8, not {dtype}")
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # have to convert to float16 or float32 first, since np.uint16 is not supported by pytorch
        dtype_np = np.float16 if dtype == torch.half else np.float32
        image_np = image_np.astype(dtype_np)
    else:
        raise ValueError(f"bit_depth must be in the range [8, 16], not {bit_depth}!")

    image = torch.from_numpy(image_np).permute(2, 0, 1).to(dtype)
    if dtype in {torch.float, torch.half}:
        image = image / (2**bit_depth - 1)

    return image

def deserialize_and_download_tensor(s3_uri: str, resource_manager: AbstractResourceManager) -> torch.Tensor:
    """Deserialize and download a tensor from S3.

    Parameters
    ----------
    s3_uri : str
        The S3 URI of the tensor.
    resource_manager : ResourceManager

    Returns
    -------
    torch.Tensor
    """
    tensor_bytes = resource_manager.get(s3_uri)
    tensor_np = zstd_decompress_arr(tensor_bytes)
    tensor = torch.from_numpy(tensor_np)
    return tensor


def test_deserialize_and_download_image():
    test_image_file = "s3://covariant-annotation-pipeline/resource_root/sim_scene_annotations/images-camera_array_01/409a/409ad6ba22b2cb129609ecbd52e5446e5f90d9920563103763e558eb576ffcf5.png"
    resource_manager = Boto3ResourceManager()
    tensor = deserialize_and_download_image(test_image_file, bit_depth=8, resource_manager=resource_manager, dtype=torch.float32)
    tensor = (tensor * 255).to(torch.uint8)
    cv2.imwrite("test_image.png", tensor.permute(1, 2, 0).cpu().numpy())

def test_deserialize_and_download_tensor():
    test_tensor_file = "s3://covariant-annotation-pipeline/resource_root/sim_scene_annotations/depth_maps-camera_array_01/fc64/fc64581dc26ef911ed77bb674b8736749351c832a5ded9d407d812da733304e9.blob"
    resource_manager = Boto3ResourceManager()
    tensor = deserialize_and_download_tensor(test_tensor_file, resource_manager)
    # Scale depth values to 0-255 range for visualization
    depth_min = tensor.min()
    depth_max = tensor.max()
    depth_normalized = ((tensor - depth_min) / (depth_max - depth_min) * 255).to(torch.uint8)
    
    # Save as grayscale image
    depth_image = depth_normalized.cpu().numpy()
    cv2.imwrite("test_depth.png", depth_image)
    print(f"Saved depth image with range [{depth_min:.2f}, {depth_max:.2f}]")


if __name__ == "__main__":
    test_deserialize_and_download_tensor()
    
