lattice_length=16
lattice_dim=1
system = dict(backbone='Liouvillian_System',
            params=dict(Lattice_length=lattice_length,
                        Lattice_dim=lattice_dim,
                        PBC=False,
                        Spin=0.5,
                        Coupling=0, #gp
                        Field_tranverse=2 #vp
                        ),
            Coupling=[0.02 * i for i in range(10)])
model = dict(backbone='NDM',
            params=dict(beta=1))
vstate=dict(backbone='MCMixedState',params=dict(n_samples=2000,
                                                n_samples_diag=512))
driver = dict(backbone='SteadyState')
work_config = dict(work_dir='./dev')
checkpoint_config = dict(load_from='E:\OneDrive\StonyBrook\QML\dev', save_inter=50)
optimizer = dict(backbone='Adam', params=dict(learning_rate=0.0001, b1= 0.9, b2 = 0.999, eps = 1e-8))
SR_conditioner = dict(enabled=True, diag_shift=0.1)
hyperpara = dict(epochs=2)