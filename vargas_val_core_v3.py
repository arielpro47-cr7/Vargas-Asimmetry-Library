import sympy as sp

class AsymmetryOperator:
    def __init__(self, precision=10):
        self.b = sp.Symbol('b', real=True) # Escala/Memoria
        self.precision = precision

    def apply(self, f, var):
        """Metodología de los 3 Pasos de Vargas."""
        f_ext = f.subs(var, var + sp.I * self.b).series(self.b, 0, self.precision).removeO()
        op_res = sp.simplify((f_ext - f) / (sp.I * self.b))
        return sp.re(op_res), sp.im(op_res)

    def get_asymmetry_ratio(self, f, var):
        """Calcula el equilibrio par/impar (Condición 50/50 del Paper 8)."""
        real_p, imag_p = self.apply(f, var)
        # En equilibrio perfecto, la magnitud de la asimetría real y el flujo imag se acoplan.
        return sp.simplify(real_p / imag_p)

    def property_regularization(self, f, var):
        """Cura de singularidades (Navier-Stokes)."""
        asym, _ = self.apply(f, var)
        return asym

    def is_autocontained(self, f, var):
        """Verifica si C=0 (Sistemas cerrados como la Función Zeta)."""
        _, imag_p = self.apply(f, var)
        # Si el límite de la parte imaginaria en escala cero es 0.
        return sp.limit(imag_p, self.b, 0) == 0

    def classify_field(self, f, var):
        """Clasificación estructural nd (Paper 4)."""
        asym, _ = self.apply(f, var)
        poly = sp.Poly(asym, self.b)
        nd = min(t[0][0] for t in poly.terms()) + 1
        return {"nd": nd, "Regime": "Classical" if nd == 1 else "Asymmetric"}
