import pyochain as pc

x = pc.Iter((1, 2, 3)).collect()
y = pc.Iter((1, 2, 3)).into(pc.Seq)
