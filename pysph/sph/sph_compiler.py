from pysph.base.ext_module import ExtModule
from pysph.sph.integrator_cython_helper import IntegratorCythonHelper
from pysph.sph.acceleration_eval_cython_helper import AccelerationEvalCythonHelper


###############################################################################
class SPHCompiler(object):
    def __init__(self, acceleration_eval, integrator):
        self.acceleration_eval = acceleration_eval
        self.acceleration_eval_helper = AccelerationEvalCythonHelper(
            self.acceleration_eval
        )
        self.integrator = integrator
        self.integrator_helper = IntegratorCythonHelper(integrator)
        self.ext_mod = None
        self.module = None

    #### Public interface. ####################################################
    def compile(self):
        """Compile the generated code to an extension module and
        setup the objects that need this by calling their setup_compiled_module.
        """
        if self.ext_mod is not None:
            return
        code = self._get_code()
        self.ext_mod = ExtModule(code, verbose=True)
        mod = self.ext_mod.load()
        self.module = mod

        self.acceleration_eval_helper.setup_compiled_module(mod)
        cython_a_eval = self.acceleration_eval.c_acceleration_eval
        if self.integrator is not None:
            self.integrator_helper.setup_compiled_module(mod, cython_a_eval)

    #### Private interface. ####################################################
    def _get_code(self):
        main = self.acceleration_eval_helper.get_code()
        integrator_code = self.integrator_helper.get_code()
        return main + integrator_code
