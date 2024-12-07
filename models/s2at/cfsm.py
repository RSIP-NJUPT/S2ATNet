import torch
import torch.nn as nn



class CFSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze1 = nn.Conv2d(2, 1, 7, padding=3)
        self.conv_squeeze2 = nn.Conv2d(2, 1, 7, padding=3)
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)  # [4,30,11,11]
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # attn = torch.cat([attn1, attn2], dim=1)
        avg_sa1 = torch.mean(attn1, dim=1, keepdim=True)
        max_sa1, _ = torch.max(attn1, dim=1, keepdim=True)
        sa1 = self.conv_squeeze1(torch.cat((avg_sa1, max_sa1), dim=1)).sigmoid()

        avg_sa2 = torch.mean(attn2, dim=1, keepdim=True)
        max_sa2, _ = torch.max(attn2, dim=1, keepdim=True)
        sa2 = self.conv_squeeze2(torch.cat((avg_sa2, max_sa2), dim=1)).sigmoid()

        attn = torch.cat((attn1 * sa1, attn2 * sa2), dim=1)
        attn = self.conv(attn)

        return x * attn



class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)  # [4,30,11,11]
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class SpatialLSKBlock3d(nn.Module):
    """
    Spatial Selection LSK Block.
    https://arxiv.org/abs/2303.09030
    """

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim // 2, 1)
        self.conv2 = nn.Conv3d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv3d(2, 2, 7, padding=3)
        self.conv = nn.Conv3d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, ...].unsqueeze(1) + attn2 * sig[:, 1, ...].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


if __name__ == "__main__":
    input = torch.randn((4, 30, 11, 11))
    lsk = CFSM(30)
    output = lsk(input)
    print(output.shape)
    pass
