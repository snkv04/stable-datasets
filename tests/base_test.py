# def test_import():
#     import aidatasets
#
# def test_CIFAR10():
#     import aidatasets
#     a = aidatasets.images.CIFAR10("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_CIFAR100():
#     import aidatasets
#     a = aidatasets.images.CIFAR100("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_Flowers102():
#     import aidatasets
#     a = aidatasets.images.Flowers102("../Downloads")
#     a.download()
#     a.load()
#
# def test_Food101():
#     import aidatasets
#     a = aidatasets.images.Food101("../Downloads")
#     a.download()
#     a.load()
#
# def test_FGVCAircraft():
#     import aidatasets
#     a = aidatasets.images.FGVCAircraft("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_IBeans():
#     import aidatasets
#     a = aidatasets.images.Beans("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_Country211():
#     import aidatasets
#     a = aidatasets.images.Country211("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_MNIST():
#     import aidatasets
#     a = aidatasets.images.MNIST("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_FashionMNIST():
#     import aidatasets
#     a = aidatasets.images.FashionMNIST("../Downloads")
#     a.download()
#     a.load()
#
#
# def test_CUB200():
#     import aidatasets
#     a = aidatasets.images.CUB200("../Downloads")
#     a.download()
#     a.load()
#
# def test_SVHN():
#     import aidatasets
#     a = aidatasets.images.SVHN("../Downloads")
#     a.download()
#     a.load()
#
# def test_RockPaperScissors():
#     import aidatasets
#     a = aidatasets.images.RockPaperScissors("../Downloads")
#     a.download()
#     a.load()
#
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--name", type=str)
#     parser.add_argument("--path", type=str)
#     args = parser.parse_args()
#
#     import aidatasets
#     a = aidatasets.images.__dict__[args.name](args.path)
#     a.download()
#     a.load()
