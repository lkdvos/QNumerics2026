### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b2b11530-a563-4e8f-b54c-4a9c419a4725
using TensorKit, TensorOperations, KrylovKit, CairoMakie, PlutoUI

# ╔═╡ 94e4d9d6-a9a2-439b-ac84-c21aabbcd386
md"""
# Tensor network primitives with TensorKit.jl

Tensor networks are built from one object, repeated and wired together: the **multi-linear map between vector spaces**.
In TensorKit.jl that object is a `TensorMap`, and this notebook introduces it from the ground up.

We work through the primitives in order:

1. **Spaces & `TensorMap`s** — vector spaces, vectors, linear maps, and tensors
2. **Index manipulation** — permuting and repartitioning legs (matricization)
3. **Contraction** — the `@tensor` macro, the working language
4. **Factorizations** — SVD and friends, with truncation
5. a **transverse-field Ising** combining everything together
"""

# ╔═╡ 1ea2749b-929d-4337-a018-2dafa6cbcf78
# fig("legend.svg")

# ╔═╡ 7dd5af3d-93a0-4bc7-84d1-1833682f18cb
fig(name) = HTML(string("<div style=\"text-align:center;margin:0.6em 0\">",
		read(joinpath(@__DIR__, "tensor_primitives", name), String), "</div>"))

# ╔═╡ f8effa9e-f7dc-4214-84fd-24719ef6892f
md"""
## 1. Vectors, matrices & vector spaces

> A tensor is something that transforms as a tensor

In order to avoid circular definitions like this, we start simple and have a look at vectors first.

### Vectors

A vector is defined as an object in a vector space, which is a set of objects that can be added, and multiplied with a scalar, in such a way that these operations are compatible with each other (associativity, distributitivy, commutativity, ...).

For the purposes of this introduction, we start with the typical vector space used in quantum physics ``\mathbb{C}^n``: (`ComplexSpace(d)`) - the complex vector space of dimension `d`.

> Both `ComplexSpace(d)` and `\mathbbC<TAB>^d` can be used to construct such vector spaces
"""

# ╔═╡ 1cb13598-8c18-4050-97a7-8b44318131aa
V = ℂ^2

# ╔═╡ 8d57fea4-546c-4d21-b7e5-db0f27403e03
W = ComplexSpace(3)

# ╔═╡ 687af255-f01b-4304-b470-0f61405b51d4
dim(V), dim(W)

# ╔═╡ a9f3f14a-54b1-4acf-853a-8c1d03e3259f
md"""
We can use these objects to define the structure of an actual vector.
For example, to obtain a vector in this vector space with random entries, we can use `rand`.
See also: `randn`, `ones`, `zeros`.
"""

# ╔═╡ 9a9d0dce-838a-418f-9183-ff8eda820d97
v = rand(V)

# ╔═╡ d56aea98-cbfe-4102-8efa-e559b1a524e2
fig("vector-box.svg")

# ╔═╡ 0eeee97a-c9cf-4604-b176-8016db3b68b9
w = ones(W)

# ╔═╡ 9703d795-e5c8-4b42-b7b6-c2ae32afddd4
md"""
Note that TensorKit does **not** hand you a flat array — it hands you a `TensorMap` whose space is

```math
\mathbb{C}^2 \;\leftarrow\; \mathbb{1},
```

For now we will ignore most of the output here, and revisit this later.
Two pieces of information should however already make sense: the printed `codomain` is the vector space we asked for, while the printed entries should correspond to the actual data we asked for.

> By default, even though we asked for a **complex** vector space, the entries of the tensors are filled with `Float64` values. You can select the element type manually by supplying it as a first argument to the tensor constructors, for example `rand(ComplexF64, V)`.

It is of course also possible to simply supply the entries manually:
"""

# ╔═╡ f3e8c698-5fa8-4188-84bf-34094a9bafc2
v2 = TensorMap([1.1, 0.3], space(v))

# ╔═╡ fbb2f571-a2c5-4932-a317-06f9127ec681
md"""
Since these are elements of a vector space, we can indeed check that they behave as vectors:
"""

# ╔═╡ 4e1a352b-e7ca-47bf-8e00-942db1e6f959
v + v2

# ╔═╡ dd0574dd-974a-4a5c-a5b9-a4d4b6043071
3 * w

# ╔═╡ cb04afcc-0a7e-4a5e-8f98-1d7db5996e39
md"""
Note that this only works for vectors that are elements of the same vector space, and you will obtain an error when combining incompatible vector spaces:
"""

# ╔═╡ eac4e3b9-bb1d-4c91-b73a-989ce3ff2b56
v + w

# ╔═╡ 26bb9675-41b8-4b18-bfc9-e3fb28e9bd04
md"""
### Matrices

A matrix (or linear map) is an operator that maps vectors to vectors.
In other words, they are maps between vector spaces.
This structure is encoded in `HomSpace`, which can be constructed as `V → W` (`\rightarrow<TAB>`).

!!! note
	Because matrices are right-multiplied with vectors `w = A * v`, TensorKit uses `W ← V` by default, since this coincides with how a matrix `A = rand(m, n)` would construct its row space `m` first, and only then its column space `n`. Both syntaxes are however accepted.
"""

# ╔═╡ ef03ea9a-888b-41ab-a662-3526739c377e
V → W

# ╔═╡ 45c52714-2a30-4603-a3ff-a551eddc54d6
A = rand(W ← V)

# ╔═╡ c133d327-cf58-42e0-9295-4b9c8e5fb4ab
fig("matrix-box.svg")

# ╔═╡ 92553163-c6e5-4633-b81e-97c212003981
A * v

# ╔═╡ ffa4defc-b6ef-4484-9649-53e18ce144d3
fig("matvec.svg")

# ╔═╡ 0a4a016e-0616-4b80-95c9-638ab8f509b1
md"""
Note how we now can start understanding slightly more of the output: a matrix maps a `domain` (column space) to a `codomain` (row space), and the size of the matrix corresponds to the sizes of the vector spaces that are involved.
"""

# ╔═╡ 98539830-cdfe-41d3-875c-a0525d2b73fd
codomain(A * v) == codomain(w)

# ╔═╡ d6a6d30c-8333-4984-9a73-74f75c1ed1ab
md"""
For some more concrete examples, we can create the Pauli operators X and Z as `TensorMap` instances:
"""

# ╔═╡ 1bafd348-959c-4c6b-9266-cf1dd33aab7b
X = TensorMap([0.0 1.0; 1.0 0.0], V ← V)

# ╔═╡ cac61c91-b7c4-495b-b192-f65c029f851f
Z = TensorMap([1.0 0.0; 0.0 -1.0], V ← V)

# ╔═╡ 77a9ae86-ac4c-4a93-ba34-9b692440f16d
md"""
Another important note is that a matrix is also a vector, in the sense that the `V → W` also is a vector space: you can add and scalar multiply matrices together, and these also obey the correct associativity and distributivity relations.
"""

# ╔═╡ d840d4f4-212f-4ca8-81c7-2b1cc740d820
md"""
### Tensors and tensor products

The tensor product is an operation that takes two vector spaces and builds a new vector space from that.
What makes it special is that this is done in a way that makes the entire construction respect multi-linearity.
It also applies to vectors, and maps a vector in ``V`` and ``W`` to one in ``V ⊗ W``.

```math
\begin{align}
(v_1 + v_2) ⊗ w &= v_1 ⊗ w + v_2 ⊗ w \\
v ⊗ (w_1 + w_2) &= v ⊗ w_1 + v ⊗ w_2 \\
λ (v ⊗ w) &= λv ⊗ w = v ⊗ λw \\
&\ldots
\end{align}
```

In TensorKit, we can mostly copy the math straight into code and verify these properties, where `\otimes<TAB>` can be used to type the tensor product symbol.
"""

# ╔═╡ e51c1c8e-3509-4147-a296-812027f9fcb8
v ⊗ w

# ╔═╡ 86b8c747-94e5-423b-9268-f3174d99c530
fig("tensor-product.svg")

# ╔═╡ 2fad41c0-19ce-4770-b490-4fefde332f34
vw = rand(V ⊗ W)

# ╔═╡ 618d38b8-7f5b-448b-a9d8-3788232068a3
md"""
Feel free to try and play around a bit with yourself this to see if you can verify the tensor product behavior.

The primary point of this construction is that a tensor product space **is** a vector space, and therefore we can simply use them in the same way.
In particular, what that means is that we can construct linear maps between tensor product spaces in the same way as before:
"""

# ╔═╡ 885414a7-b1fe-4edf-a0ba-7a5a51b3983f
B = rand(W ⊗ V ← V ⊗ W)

# ╔═╡ 858ae02e-f54e-4498-a649-2e692fd2cf1f
fig("tensormap-general.svg")

# ╔═╡ 631e2384-66d3-43ed-80ff-4d9b1ee28a31
B * vw

# ╔═╡ b50d37ce-fefa-4934-a5f0-8786a4da1703
md"""
Finally, this also means that we can recursively extend this definition to tensor products of an arbitrary number of vector spaces, and linear maps between any combination of numbers of spaces.

In TensorKit, all these objects are represented as objects of type `TensorMap{T, S, N1, N2, A}`.
Here, we have:
- `T`: the `scalartype` or `eltype` of the entries
- `S`: the `spacetype` of the tensor (see symmetries)
- `N1`: the `numout`, number of factors in the `codomain`
- `N2`: the `numin`, number of factors in the `domain`
- `A`: the `storagetype`, which can be used to store tensors on GPU devices

Note that these values can all be accessed through those functions:
"""

# ╔═╡ 89d563ea-8929-4ab0-9d15-ec59c0f90fe5
numin(B), scalartype(v), numout(B * vw)

# ╔═╡ 71ce2788-1a86-4ab4-81e5-08f16caeb53b
md"""
### Tensors as multi-linear maps

As hinted by the fact that these are all represented by objects of the same type `TensorMap`, we can now unify all these definitions and come to our final definition of a tensor:

> A tensor is a multilinear map from a tensor product of `numin` vector spaces to a tensor product of `numout` vector spaces.

In particular, a vector is typically reserved for the special case `numout(v), numin(v) == (1, 0)` and a matrix for the special case `numout(A) == numin(A) == 1`.
However, **any** tensor is a vector in the sense that it is an element of a vector space, and **any** tensor is a matrix in the sense that it is a map between vector spaces.
This equivalences are exactly what we will be using in the following sections, and we will just mix and match and pick the one that is most convenient to us.
"""

# ╔═╡ 9524e71a-edbf-4118-84bb-0209698dcfe6
fig("vec-mat-tensor.svg")

# ╔═╡ 5f58e0b3-e941-4694-98ec-0cb76b93ca13
md"""
## 2. Index manipulations

A `TensorMap` carries two pieces of bookkeeping at once: *which* vector spaces sit
on its legs, and *how* those legs are split into a **codomain** (the output spaces,
written on the left of `←`) and a **domain** (the input spaces, on the right). In
§1 we saw that this split is what decides whether a tensor reads as a "vector"
(everything in the codomain) or as a "matrix" / linear map (codomain ``\leftarrow``
domain).
In this section we learn how to change that bookkeeping, which is the base of most of the algorithmic primitives that follow.

### Permutations

While we cannot add vectors in `V ⊗ W` with ones in `W ⊗ V`, there is actually a way to relate these two vector spaces together.
In particular, this is something you may have already seen before, as this is what `Base.permutedims` is doing.

In TensorKit, we can achieve the same thing with the convenient `@tensor` macro.
Remember that a macro is a piece of code that transforms code into other code.
In this particular case, the code that is being transformed is a special flavor of index notation, where we give labels to each of the indices of a tensor:
"""

# ╔═╡ f229f3c8-ec88-4b59-96d6-9693a1d3c839
let
	t1, t2 = rand(V ⊗ W), rand(W ⊗ V)
	t1 + t2 # error because of incompatible spaces
end

# ╔═╡ 12d8f0e6-3cbe-4543-90b3-491561c3010f
let
	t1, t2 = rand(V ⊗ W), rand(W ⊗ V)
	@tensor t2_permuted[b a] := t2[a b]
	t1 + t2_permuted
end

# ╔═╡ 7c0e0c02-893e-4e80-9741-6869a7c826ed
fig("permutation.svg")

# ╔═╡ 23c24e86-25ac-4b94-a4da-94773e491c96
md"""
Note that the labels `a` and `b` are arbitrary, and are just used to identify spaces on the left and spaces on the right-hand side.
You can equally-well use longer labels `mylonglabel`, or integers `1`, `2`, ...

Also note that we used the symbol `:=` instead of simply using `=`.
The former will create a **new** tensor and fill it with the permuted data, whlie the latter will fill the contents of an **existing** tensor and fill it with the permuted data, which is sometimes useful to avoid having to allocate new data.

Finally, note that also `+=` and `-=` are supported, and on top of that the `@tensor` macro will recognize arbitrary linear combinations:
"""

# ╔═╡ c522c436-266e-4c82-8f30-93ddfc3ba1db
let
	t1, t2 = rand(V ⊗ W), rand(W ⊗ V)
	@tensor t3[a b] := 0.2 * t1[a b] + t2[b a]
end

# ╔═╡ 3cd0da64-769e-42f9-8b37-20da133d42aa
md"""
### Repartition

Similarly, we can relate `W ← V` with `W ⊗ V'`, where we now introduce the notion of a dual vector space `V'`.
A dual space `V'` is defined as the vector space of linear functions on `V`, which is why and how `W ← V` is **isomorphic** to `W ⊗ V'`.
For vector spaces without structure, the isomorphism is actually completely trivial, and with a slight abuse of notation, we can simply say the two are equal.

This is in fact something that we are used to already in the context of quantum mechanics, where it is common to write an operator ``O`` as
```math
O ≡ O_{ij} |i⟩⟨j|
```
where we write ``|i⟩`` for the ket (vector), and ``⟨j|`` for the bra (covector or dual vector).

The `@tensor` macro handles this by a slight adaptation of the notation.
Instead of `t[a b c]`, we can use `t[a b; c]` to denote the partition into domain and codomain.

!!! note
	For convenience, this is only really enforced on the left-hand side of the `@tensor` equations, and the right-hand side will simply accept the labels in their linear order (first enumerating the codomain, followed by the domain spaces)
"""

# ╔═╡ e1f2adf3-df9b-4094-9cad-de71cabf6e3e
let M = TensorMap(collect(1:dim(W ⊗ V ← W)), W ⊗ V ← W)
	@tensor M2[a b c] := M[a b; c]
	@tensor M3[a; b c] := M[a b c] # rhs ignores ;
	@tensor M4[(); a b c] := M[a b; c] # () for empty codomain 
	[space(M2), space(M3), space(M4)]
end

# ╔═╡ 5c0561e9-1747-4069-8c10-72d26b7c2e9b
fig("repartition.svg")

# ╔═╡ 1a310857-f3fc-4547-b0c0-c411af51e630
md"""
### Matricization

The key point that resurfaces again is that choosing the split between codomain and domain is completely equivalent to matricization.
Interpreting the codomain (domain) as a whole as a vector space, we end up with the matrix representation for the resulting linear map, which is automatic for `TensorMap`s.
So whenever we collapse all the codomain legs into a single combined row index and all the domain legs into a single column index, we have turned our order-``N`` tensor into an honest ``(\text{codomain dim}) \times (\text{domain dim})`` matrix.
*Which* legs we send up versus down decides *which* matrix we get.

For ``M \colon W ⊗ V ← W`` (with dimensions ``3, 2, 3``) the repartitions give matrices of genuinely different shapes, for example:

```math
\underbrace{(ℂ^3\otimes ℂ^2)}_{\text{6 rows}} \leftarrow \underbrace{ℂ^3}_{\text{3 cols}}
\qquad\text{vs.}\qquad
\underbrace{ℂ^3}_{\text{3 rows}} \leftarrow \underbrace{(ℂ^3\otimes ℂ^2)}_{\text{6 cols}} .
```
"""

# ╔═╡ 5341c0e2-eb9f-4d14-874a-98c26742cecb
fig("matricization.svg")

# ╔═╡ 52ef36df-f215-4ded-98f8-f9b68b9b0a91
M = TensorMap(collect(1:18), W ⊗ V ← W)

# ╔═╡ 67d8e3ee-21f3-4f94-af11-5061fd24c77c
@tensor M2[a; b c] := M[a b; c]

# ╔═╡ 8f87cda8-e656-4fab-8c1b-305126530def
md"""
Each split is a different *cut* through the tensor, and each gives a different
matrix to factorize. When we feed a tensor to an SVD in §4 we are really choosing
one of these cuts and asking how the legs on the left correlate with the legs on
the right — exactly the "split the system into left and right, then look at the
singular values" picture from the low-rank compression demo.

Note that this really **is** reshaping the data, and is not actually doing any different manipulations on that data.
It is just relabeling which elements are grouped.
This can be seen from comparing `M` and `M2`, which is exactly how `Base.permutedims` and `Base.reshape` would behave on regular arrays.
"""

# ╔═╡ f3265508-02cd-4a53-8648-941ddd2bf5cc
md"""
## 3. Tensor contractions

Next up is the operation that allows us to evaluate networks of tensors, called tensor contraction.

```math
C_{ijk…} = A_{ijabc…} * B_{abk…}
```

In principle, we have already encountered all components that are needed to properly define this: there is `*` for evaluating a linear map, and `@tensor` to permute and matricize the indices such that everything lines up for that matrix multiplication.
In fact, this is often the most performant way of evaluating the tensor contraction to begin with, since it allows us to use dedicated kernels for these operations that are often hand-crafted for extreme high-performance (BLAS).

The `@tensor` macro actually lets us bypass the need to manually permute and matricize, and instead will automatically compile down to the most efficient implementation.
The convention for the notation is called Einstein summation convention, which just means that all labels in the expression appear twice:
- either they are both on the right-hand side of the equation, indicating summation
- alternatively they appear once on the left-hand side and once on the right-hand side, in which case they indicate open indices that are permuted.

The remainder of the `@tensor` machinery is of course still available:
"""

# ╔═╡ d52cf845-a2f1-45db-b6c6-eba149df00cb
fig("contraction.svg")

# ╔═╡ 939dab51-20f9-4d76-9fac-4ce94f836ae2
let A = rand(W, V), v = rand(V)
	@tensor w[a] := A[a; b] * v[b] # matrix multiplication
	@tensor conj(v[a]) * v[a] # inner product
	@tensor vw[a b] := v[a] * w[b] # outer product
	# ...
end;

# ╔═╡ 069866ec-8ca1-49d1-bf9e-c98652964529
fig("inner-outer-matmul.svg")

# ╔═╡ 30f71f8b-b0ee-41fe-87be-0d15a9d1d2e9
md"""
A **trace** is the same idea with both legs on a single tensor: repeat a label
on one box and it gets summed, ``\sum_i A_{ii}``. With no surviving labels the
result is a scalar, so we drop the output bracket entirely and write:
"""

# ╔═╡ 15ec7fc0-a50e-44e8-ba41-8cc918c8f5b5
fig("trace.svg")

# ╔═╡ 3583a901-f478-4ef3-a09b-fc6e6049fe14
let A = rand(V ← V)
    t1 = @tensor A[i; i]
    t1, tr(A)
end

# ╔═╡ efab999d-2ccd-4002-83b4-2173de42e37f
md"""
### Contraction order

We can extend this functionality to contract arbitrary collections of tensors, by reducing them to a sequence of pairwise contractions.
The `@tensor` macro handles this automatically, and will simply pairwise contract from left-to-right.

The actual order of the pairwise sequence does not actually affect the result (up to rounding errors of floating point operations), which you can easily check:
"""

# ╔═╡ e204c1e0-318f-466e-89b4-823391c7cf58
let
	@tensor XZv1[a] := X[a; b] * Z[b; c] * v[c]
end

# ╔═╡ 24bacdfd-ba40-4ac4-9a4b-586950be20e2
let
	@tensor XZv2[a] := Z[b; c] * v[c] * X[a; b]
end

# ╔═╡ 61ff9b19-6ddc-472f-90b7-7222770932e9
md"""
Nevertheless, the number of operations is actually affected quite heavily.
To see this, we can make the tensors slightly bigger and measure:
"""

# ╔═╡ 1cf78c09-abe4-4ef1-be3d-d29c0b117a1f
let n = 1_000
	V = ComplexSpace(n)
	A = rand(V, V)
	v = rand(V)
	f(A, v) = @tensor AAv[a] := A[a; b] * A[b; c] * v[c]
	g(A, v) = @tensor AAv[a] := A[a; b] * (A[b; c] * v[c])
	f(A, v), g(A, v) # run once to avoid measuring compilation time
	@time f(A, v)
	@time g(A, v)
end;

# ╔═╡ d70f5db3-ade2-400e-9df8-83e2d2ca5169
md"""
Can you explain what is going on here?
As a hint, you can try and count the number of scalar multiplications that are being carried out.
You should be able to show that `f` scales as ``\mathcal{O}(n^3)``, while `g` scales as ``\mathcal{O}(n^2)``.
"""

# ╔═╡ 3f17f716-ee74-4666-a292-d962519a1a70
fig("contraction-order.svg")

# ╔═╡ 42a1ace4-9d95-490c-82a1-06321afe1013
md"""
### `ncon`: the numbered-index sibling

In the example above we already showed that you can use parentheses to control the order of operations in a `@tensor` expression.
However, the community has come up with a convention to specify the contraction order in a slightly more convenient way.
In particular, the so-called `ncon`-style syntax: 

- **positive** integers mark contracted legs — each value appears exactly
  **twice**, once on each tensor it joins, contraction happens in ascending order.
- **negative** integers mark the **output** legs, and they are ordered by
  decreasing value, i.e. `-1` is the first output leg, `-2` the second, and so
  on (most-negative last).

For example, the following code is equivalent to the previous block, without parentheses:
"""

# ╔═╡ 084e8438-5641-4321-80f8-15d27b581040
fig("ncon.svg")

# ╔═╡ 5887259a-11c6-497a-b25c-ac8af98fa41b
let n = 1_000
	V = ComplexSpace(n)
	A = rand(V, V)
	v = rand(V)
	f(A, v) = @tensor AAv[-1] := A[-1; 1] * A[1; 2] * v[2]
	g(A, v) = @tensor AAv[-1] := A[-1; 2] * A[2; 1] * v[1]
	f(A, v), g(A, v) # run once to avoid measuring compilation time
	@time f(A, v)
	@time g(A, v)
end;

# ╔═╡ ae357eba-748e-4cc5-a9de-906781af124f
md"""
For completeness, note that it can sometimes be usefl to have dynamically generated expressions.
For this, the `ncon(tensors, indices)` function can be used to evaluate expressions that can be built programmatically, and the following expression is again the same:
"""

# ╔═╡ 6f896e6f-bf9a-40aa-bde2-8fe3a462f9bb
let n = 1_000
	V = ComplexSpace(n)
	A = rand(V, V)
	v = rand(V)
	g(A, v) = @tensor AAv[-1] := A[-1; 2] * A[2; 1] * v[1]
	h(A, v) = ncon([A, A, v], [[-1, 2], [2, 1], [1]])
	g(A, v), h(A, v) # run once to avoid measuring compilation time
	@time g(A, v)
	@time h(A, v)
end;

# ╔═╡ e41b227e-0d68-4eca-9055-3c479f39a22a
md"""
!!! note
	Since the exact network for `ncon` has to be evaluated at runtime (it depends on the value of `network`, not only the type), `ncon` is not a type-stable function. Often, the tensor contractions themselves take enough time for this to not matter too much, but in high-performant scenarios it can be beneficial to avoid this.
"""

# ╔═╡ 7a106f3a-9ea1-4a25-a044-26633ef16604
md"""
## 4. Factorizations & truncated factorizations

Similar to tensor contractions, we can define matrix factorizations for tensors by *matricizing* or partitioning the input tensor correctly, and then simply factorizing the resulting matrix representation.
"""

# ╔═╡ 09ab54ba-6d99-4449-a1f6-06f8cee09b75
md"""
### Singular value decompositions

The **singular value decomposition** factorizes a matrix as ``m = U S V``, with ``U`` and ``V`` isometric (``U^\dagger U = \mathbb{1}``, ``V V^\dagger = \mathbb{1}``) and ``S`` a non-negative diagonal of singular values.
Since `M` is already a matrix once we read it as the map ``W ⊗ V ← W``, we can hand it straight to `svd_compact`, which returns the triple ``(U, S, V)``.
The **bond space** that appears in the middle — the codomain/domain of ``S`` — is the column space of the chosen cut.
"""

# ╔═╡ a2b5fe58-2ed0-43b2-80c2-a8b044237969
fig("svd-chain.svg")

# ╔═╡ 0a465a6e-c223-42fa-9eb1-48a198d2aeb0
Usvd, Ssvd, Vsvd = svd_compact(M);

# ╔═╡ 4f46a66a-5c29-4445-8b94-d7f72099f5a8
Usvd * Ssvd * Vsvd ≈ M, space(Ssvd)

# ╔═╡ c8a63111-d3aa-4c7f-98d3-6e031b9a628d
md"""
Remember that the factorization is of *a chosen cut*: a different matricization gives a different SVD through a different bond space.
For the common case where we only want the singular values, `svd_vals` returns them directly, and `svd_full` adds the zero block to give a square ``S`` and full unitary ``U`` and ``V``.
"""

# ╔═╡ 14b9d4a2-1c80-4637-bd88-fa7eaabc43f7
σs = collect(svd_vals(M))

# ╔═╡ 71441b76-d49d-4926-9e56-20fb57fc4dd4
md"""
The reason singular values matter so much is **truncation**: keeping only the largest ``k`` of them gives the best rank-``k`` approximation of the matrix (the Eckart–Young theorem).
`svd_trunc` does exactly this, taking a *truncation strategy* such as `truncrank(k)`.
Note that it returns a **4-tuple** ``(U, S, V, ϵ)``, where the trailing ``ϵ`` is the truncation error.

The relative error is not arbitrary — it is governed exactly by the singular values that were thrown away:

```math
\frac{\lVert m - m_k\rVert}{\lVert m\rVert} = \sqrt{\frac{\sum_{i>k}\sigma_i^2}{\sum_i \sigma_i^2}} ,
```

which is precisely the discarded-weight formula from the low-rank image-compression demo, now applied to a bond of a tensor.
This is the same quantity that will control the truncation error in TEBD and DMRG.
"""

# ╔═╡ f3a38a7c-b29f-4c21-81cc-b8d7b114f9d4
fig("svd-truncation.svg")

# ╔═╡ 6eb05f47-4e80-491a-a5df-d6a90e5d200d
Ut, St, Vt, ϵ = svd_trunc(M; trunc = truncrank(1));

# ╔═╡ 27d4b849-26d6-4ece-9de5-0b2f35fc9470
norm(Ut * St * Vt - M) / norm(M), sqrt(sum(abs2, σs[2:end]) / sum(abs2, σs))

# ╔═╡ ada68578-e35d-448b-8ed1-35fbb61e1336
md"""
### Eigenvalue decompositions

When the codomain and domain are the *same* space — an endomorphism, like our Pauli operators or a Hamiltonian — we can diagonalize instead.
`eig_full` handles the general case (complex eigenvalues), while `eigh_full` specializes to the **hermitian** case, where the eigenvalues are real and the eigenvectors orthonormal.

As an example, take the (hermitian, positive-semidefinite) map ``G = A^\dagger A``, and check that ``G = U D U^\dagger``:
"""

# ╔═╡ 9160eb93-bc34-4c9e-8abb-1f0baec13be1
G = A' * A

# ╔═╡ 974b9dc4-ab44-48ce-accd-da102da5cbc6
Deig, Ueig = eigh_full(G);

# ╔═╡ 6cbe0a83-6645-4eb2-a323-8c89945c829c
Ueig * Deig * Ueig' ≈ G

# ╔═╡ 1fc05087-9a65-4113-bb1c-87b7023ec5cb
md"""
This connects the two factorizations directly: the eigenvalues of ``A^\dagger A`` are exactly the *squares* of the singular values of ``A``.
We will lean on this identity in the next section, where the eigenvalues of a reduced density matrix are the squared Schmidt coefficients.
"""

# ╔═╡ db70cf96-c30d-44a7-9194-5ac7108376e1
fig("svdvals-eig.svg")

# ╔═╡ 3245e811-8157-472d-9311-34654101b11e
sort(collect(eigh_vals(G))) ≈ sort(collect(svd_vals(A)) .^ 2)

# ╔═╡ 61ccf0a8-04b0-4e80-8d53-042acb1c3526
md"""
### Orthogonal factorizations

Often we don't need the full singular value spectrum, but only want to **gauge** a bond — to replace a tensor by an *isometry* times a small remaining factor.
This is the workhorse for bringing a matrix product state into canonical form.
`left_orth` gives a QR-style ``(Q, R)`` with ``Q^\dagger Q = \mathbb{1}``, so the codomain legs become orthonormal; `right_orth` gives an LQ-style ``(L, Q)`` with ``Q Q^\dagger = \mathbb{1}`` instead.
Both reconstruct the original tensor exactly.
"""

# ╔═╡ 6778a8a9-8ed2-4650-81ff-68b158ef05bf
fig("isometry.svg")

# ╔═╡ 5e8ea935-fd55-4d1d-a9e3-4ce6c7e0194f
Ql, Rl = left_orth(M);

# ╔═╡ 93e32ca6-d730-4278-ad3d-2b032ca9c566
norm(Ql' * Ql - id(domain(Ql))), Ql * Rl ≈ M

# ╔═╡ be0b4b7b-e7b7-40e5-8b0e-1077c6a536b4
Lr, Qr = right_orth(M);

# ╔═╡ 0b1781be-9e9d-4eb1-b4da-cf289105c743
norm(Qr * Qr' - id(codomain(Qr))), Lr * Qr ≈ M

# ╔═╡ 158afe28-41b4-4834-a0e8-ec0505bf69b5
md"""
## 5. The transverse-field Ising model

Time to put every primitive together on a real physics problem.
For that, we consider the [Transverse Field Ising Model](https://en.wikipedia.org/wiki/Transverse-field_Ising_model).
The model lives on a chain of ``N`` spin-``\tfrac12`` sites.
Its Hamiltonian is defined as:

```math
H \;=\; -J\sum_{i=1}^{N-1} Z_i Z_{i+1}\;-\;g\sum_{i=1}^{N} X_i .
```

We will build ``H`` as **one big `TensorMap`** acting on the full Hilbert space ``\mathbf{d}^{\otimes N}``, find its ground state with a Krylov eigensolver, and then read off two physical quantities — the magnetization and the entanglement spectrum.
This is the *dense* reference solver: it works directly on the full ``2^N``-dimensional space, with no tensor-network structure, where everything stays a `TensorMap`.
The ground state energy ``E_0`` it produces is the target that every later method (TEBD, DMRG) must reproduce.

!!! warning
	Since this is the dense reference, the computational cost scales exponentially with the system size. Most laptops will probably be able to handle 8 qubits just fine, but things can get dramatically more expensive as `N` increases.
"""

# ╔═╡ c1c715d5-fc99-44d8-ad63-ca3702d4fcc6
md"""
First, the local building blocks. The Pauli operators and the identity are just ``2\times 2`` matrices, promoted to single-site maps ``\mathbf{d}\leftarrow\mathbf{d}``.
We already defined `X` and `Z`, so we simply need to define the `N`-dimensional operators.
"""

# ╔═╡ d05ebb75-976d-4919-bf4f-2d6a02eee150
d = ℂ^2

# ╔═╡ 3b8c3cf1-4790-4775-8ca6-903ccf96d87e
md"""
Use the sliders to set the system size and couplings — every result below recomputes reactively (keep `N` modest, the cost grows like ``2^N``):

- `N` = $(@bind N PlutoUI.Slider(2:12; default = 8, show_value = true))
- `J` = $(@bind J PlutoUI.Slider(0.0:0.1:2.0; default = 1.0, show_value = true))
- `g` = $(@bind g PlutoUI.Slider(0.0:0.1:2.0; default = 1.0, show_value = true))
"""

# ╔═╡ 3321bb19-a246-4992-9991-799ca22b623e
I = id(d)

# ╔═╡ 4f47cb0f-98c7-4013-9dbf-f76052cadffa
"""
    X_i(i, N)

Create the operator `X_i` acting as `X` on qubit `i`, and as `I` on all other `N` qubits. 
"""
function X_i(i, N)
    operators = fill(I, N)
    operators[i] = X
    return reduce(⊗, operators)
end

# ╔═╡ 180c9636-7400-4870-a7fb-e5665d6b1767
"""
    Z_iZ_j(i, j, N)

Create the operator `Z_iZ_j` acting as `Z` on qubit `i` and `j`, and as `I` on all other `N` qubits. 
"""
function Z_iZ_j(i, j, N)
    operators = fill(I, N)
    operators[i] = operators[j] = Z
    return reduce(⊗, operators)
end

# ╔═╡ 66d6fbf6-8d72-4c5b-9a82-b0c0c17be1c2
fig("local-ops.svg")

# ╔═╡ 24a36817-886b-4555-8982-a5bc68c10bd5
"""
	ising_hamiltonian(N; J = 1.0, g = 1.0)

Create the tensor map that represents the Hamiltonian for the transverse field Ising model, defined as:

```math
	H = -J ∑ᵢ ZᵢZᵢ₊₁ - g ∑ᵢ Xᵢ
```
"""
function ising_hamiltonian(N; J = 1.0, g = 1.0)
	return -J * sum(i -> Z_iZ_j(i, i + 1, N), 1:N-1) - g * sum(i -> X_i(i, N), 1:N)
end

# ╔═╡ dbbe770e-4149-4bd1-bdea-dc52d13b3f11
H = ising_hamiltonian(N; J, g)

# ╔═╡ 489242de-ad70-46c2-ad7d-f0fbd1e3f6f2
fig("full-h.svg")

# ╔═╡ ec0cad09-b01e-454a-bd14-bc01538b588b
md"""
### The ground state

For a small system, it is possible to simply diagonalize the entire Hamiltonian, and find the lowest eigenvalue that way.
However, we can be slightly more efficient here since we are only interested in the ground state of the system and use a Krylov method instead.
Since we will be using this as well later, we introduce that here as well.

In Julia, this is quite easy and we can simply make use of generic packages such as [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).
This package exposes the `eigsolve` function to compute (at least) `howmany` eigenvalue-vector pairs:

```julia
λs, vecs, info = eigsolve(f, x0, howmany, which; kwargs...)
```

In this case, `which = :SR` is used for the smallest real value, `f` is a function handle that implements the action of the operator, and `x0` is an initial guess for the ground state.
Since we know our Hamiltonian is hermitian, we can additionally use `ishermitian = true` as one of the keyword arguments to use a more efficient solver.
"""

# ╔═╡ d1366be4-8877-4a10-993e-21bfdddbe22a
psi0 = rand(d^N);

# ╔═╡ 8fe61599-26df-4e1b-a75b-009fc43486d2
begin
	Es, psis, info = eigsolve(psi0, 1, :SR; ishermitian = true) do psi
		return H * psi
	end
	groundstate = first(psis)
	E0 = first(Es)
end

# ╔═╡ 84ccbf05-bd50-4348-875f-87ae372a0e8b
Markdown.parse("""
With ``N = $(N)`` and ``J = g = 1``, the dense solver gives a ground-state energy

```math
E_0 = $(round(E0; digits = 6)), \\qquad E_0 / N = $(round(E0 / N; digits = 6)).
```
""")

# ╔═╡ 6eefd53d-c001-4696-bf83-039953c19c83
md"""
### Magnetization

A ground-state expectation value ``\langle O\rangle = \langle\psi|O|\psi\rangle`` is just an inner product after applying the operator, for which Julia exposes: `dot(psi, O, phi)`.
For the transverse magnetization on site ``i`` we place an ``X`` there (identities elsewhere) and contract against the state.
"""

# ╔═╡ eb8e42b4-e704-488c-80e8-d9afbc6bc162
magnetization(psi, i, phi = psi) = real(dot(psi, X_i(i, N), phi))

# ╔═╡ c7c85315-ff3c-4c69-b1f8-3ff0b077cb86
fig("expectation.svg")

# ╔═╡ 30960ab5-4c3c-4cdb-a8c8-0335011b2864
Ms = magnetization.(Ref(groundstate), 1:N)

# ╔═╡ a310bf9c-8b9c-482f-9c4c-76551ea6bc27
let
    fig = Figure(; size = (640, 360))
    ax = Axis(fig[1, 1]; xlabel = "site i", ylabel = "⟨Xᵢ⟩", xticks = 1:N,
        title = "Transverse magnetization across the chain")
    scatterlines!(ax, 1:N, Ms; color = :crimson)
    fig
end

# ╔═╡ ece8ecc2-3a50-4c04-b314-50fa532f21f8
md"""
### Entanglement spectrum

Finally, let's actually evaluate the statements about **Schmidt coefficients** and entanglement spectra.
These are defined by splitting up the total system into two subsystems, and measuring the entanglement between them.
We can do this explicitly: we take the ground state and construct the reduced density matrix, and then simplify using `eig_vals(A'A) = svd_vals(A).^2`.

Reinterpreting the ground-state vector as a matrix across the cut between the subsystems is exactly a `repartition`: we send the first ``N/2`` legs to the codomain and the rest to the domain, and then compute the singular values.

These singular values are the **Schmidt coefficients** ``\sigma_i`` of the bipartition.
Their squares (normalized) are the probabilities ``p_i = \sigma_i^2 / \sum_j\sigma_j^2``, and the **entanglement entropy** is

```math
S \;=\; -\sum_i p_i \log p_i .
```

A rapidly decaying Schmidt spectrum means the state is *close to low rank across the cut*, or equivalently the entanglement is low.
This is precisely the property that will let a matrix product state represent it with a small bond dimension.
"""

# ╔═╡ ac718414-7df8-4d43-950f-bbe3a7831a6c
fig("bipartition.svg")

# ╔═╡ 7f696194-bcfc-4b46-8974-4cbebde2fd7c
# Schmidt coefficients σᵢ: the singular values across the central cut.
Σ = sort(collect(svd_vals(repartition(groundstate, N ÷ 2))); rev = true)

# ╔═╡ cad5d22b-4ab1-423b-84de-8e142568ddbc
# Born probabilities pᵢ = σᵢ² / Σⱼ σⱼ².
ps = normalize(Σ .^ 2, 1)

# ╔═╡ 3b4561f9-f95d-4294-83ea-6ebf59ba1243
safe_x_log_x(x) = x > 0 ? x * log(x) : zero(x)

# ╔═╡ 31ee3fbe-f70f-4e94-b952-d455ec9564a6
# Entanglement entropy S = -Σ pᵢ log pᵢ (terms with pᵢ = 0 contribute nothing).
S_ent = -sum(safe_x_log_x, ps)

# ╔═╡ 80847bbf-eb87-4d1e-a372-13ccb91131eb
let
    pos = filter(>(1e-15), Σ)   # log axis: keep the numerically non-zero coefficients
    fig = Figure(; size = (640, 360))
    ax = Axis(fig[1, 1]; xlabel = "index i", ylabel = "σᵢ", yscale = log10,
        title = "Entanglement (Schmidt) spectrum across the central cut")
    scatter!(ax, 1:length(pos), pos; color = :steelblue)
    fig
end

# ╔═╡ 5de28869-1826-40b6-9258-7a9c4297e0bd
Markdown.parse("""
The first few Schmidt coefficients across the central cut are
``$(join(round.(Σ[1:min(4, end)]; digits = 4), ",\\; "))``,
and they fall off fast — only a handful matter. The corresponding
entanglement entropy is

```math
S = $(round(S_ent; digits = 4)).
```
""")

# ╔═╡ bcf25e24-5646-4590-966f-7ca0b1ca41b8
md"""
## 6. Sweeping the transverse field

The transverse field `g` drives a **quantum phase transition** at ``g = J``. For ``g < J`` the Ising coupling wins and the ground state is ferromagnetically ordered; for ``g > J`` the field wins and polarizes every spin along ``x``. We can watch the crossover by re-solving the dense ground state for a range of `g` and reading off the average transverse magnetization ``\langle X\rangle / N`` and the half-chain entanglement entropy ``S``.

- The **magnetization** climbs from ``0`` (ordered) toward ``1`` (fully polarized), with its steepest change around the critical point ``g = J``.
- The **entanglement entropy** sits near ``\ln 2`` throughout the ordered phase — the exact finite-size ground state there is the symmetric "cat" state ``|{\uparrow\cdots}\rangle + |{\downarrow\cdots}\rangle``, worth one bit across any cut — and then collapses toward zero once the field polarizes the chain into a near-product state.

This sweep re-runs the dense solver once per point, so it uses the slider values of `N` and `J` but scans `g` itself; the dashed line marks the current slider `g`.
"""

# ╔═╡ 2dfb7a0e-e80e-4d63-93c7-8432e4c0c605
begin
    g_values = range(0, 2; length = 21)
    mags_vs_g = Float64[]
    Ss_vs_g = Float64[]
    for gᵢ in g_values
        Hᵢ = ising_hamiltonian(N; J, g = gᵢ)
        _, vecs, _ = eigsolve(ψ -> Hᵢ * ψ, rand(d^N), 1, :SR; ishermitian = true)
        ψ₀ = first(vecs)
        push!(mags_vs_g, sum(i -> magnetization(ψ₀, i), 1:N) / N)
        σ = collect(svd_vals(repartition(ψ₀, N ÷ 2)))
        push!(Ss_vs_g, -sum(safe_x_log_x, normalize(σ .^ 2, 1)))
    end
end

# ╔═╡ 86577720-cb91-4118-a561-31b082165e50
let
    fig = Figure(; size = (640, 480))
    ax1 = Axis(fig[1, 1]; ylabel = "⟨X⟩ / N",
        title = "TFIM across the transition  (N = $N, J = $J)")
    ax2 = Axis(fig[2, 1]; xlabel = "transverse field g", ylabel = "entanglement entropy S")
    scatterlines!(ax1, g_values, mags_vs_g; color = :crimson)
    scatterlines!(ax2, g_values, Ss_vs_g; color = :steelblue)
    vlines!(ax1, [g]; color = :gray, linestyle = :dash)
    vlines!(ax2, [g]; color = :gray, linestyle = :dash)
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid = false)
    rowgap!(fig.layout, 6)
    fig
end

# ╔═╡ f0284265-e6ab-4b32-89b4-2c67de4f87c1
md"""
For those in the know, the TFIM should have a second order quantum phase transition at ``g = J``.
In particular, this means that we expect discontinuous jumps in the expectation values, and a diverging entanglement entropy.
In order to actually demonstrate this, we however need to push the system sizes beyond what is accessible with exact-diagonalization methods, since this only happens in the thermodynamic limit and ``N ∼ 10`` is simply too far off.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
TensorKit = "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"
TensorOperations = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"

[compat]
CairoMakie = "~0.15.12"
KrylovKit = "~0.10.3"
PlutoUI = "~0.7.83"
TensorKit = "~0.17.0"
TensorOperations = "~5.6.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "17775c7c0abeaa99e874e88f450fc12b8fa2e3d0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "2eeb2c9bef11013efc6f8f97f32ee59b146b09fb"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.44"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7715e5b2b186c4d9b664d299d2c9e48b9a778c88"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.6.1"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "80b2770813b42f80235ea57f4333de8ff3e1c342"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.12"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkSplitters]]
git-tree-sha1 = "1c52c8e2673edc030191177ff1aee42d25149acb"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.2.0"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "7bc84b769c1d384315e7b5c4ac03a6c303e6cf35"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.8"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "a55766a9c8f66cf19ffcdbdb1444e249bb4ace33"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.6"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3c8a0a9a6d4a10bdfb6b751bd2b6051ed3e25fd4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.127"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "364685f5ffde25deb1bbcfd5bb278a5c6b7a9b37"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.11"

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

    [deps.GeometryBasics.weakdeps]
    Extents = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.HalfIntegers]]
git-tree-sha1 = "9c3149243abb5bc0bad0431d6c4fcac0f4443c7c"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.6.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "921d7e91687e15a2c7c269c226960491fc041832"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.9"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.KrylovKit]]
deps = ["LinearAlgebra", "PackageExtensionCompat", "Printf", "Random", "VectorInterface"]
git-tree-sha1 = "1ab4539f47bee7e5045f679e5136ccaa3c8c40c3"
uuid = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
version = "0.10.3"
weakdeps = ["ChainRulesCore"]

    [deps.KrylovKit.extensions]
    KrylovKitChainRulesCoreExt = "ChainRulesCore"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "efe001e1ee81b8eee0fe7da5a4328fcbbfd6b3aa"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.12"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.MatrixAlgebraKit]]
deps = ["LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "db7139ec5c2bd2c4f68aa750793cb032b600673d"
uuid = "6c742aac-3347-4629-af66-fc926824e5e4"
version = "0.6.8"

    [deps.MatrixAlgebraKit.extensions]
    MatrixAlgebraKitAMDGPUExt = "AMDGPU"
    MatrixAlgebraKitCUDAExt = "CUDA"
    MatrixAlgebraKitChainRulesCoreExt = "ChainRulesCore"
    MatrixAlgebraKitEnzymeExt = "Enzyme"
    MatrixAlgebraKitGenericLinearAlgebraExt = "GenericLinearAlgebra"
    MatrixAlgebraKitGenericSchurExt = "GenericSchur"
    MatrixAlgebraKitMooncakeExt = "Mooncake"

    [deps.MatrixAlgebraKit.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    GenericLinearAlgebra = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
    GenericSchur = "c145ed77-6b09-5dd9-b285-bf645a82121e"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "9a07c25c438110500d871fd5309649ec6791ef57"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.6"
weakdeps = ["Markdown", "ProgressMeter"]

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"
    ProgressMeterExt = "ProgressMeter"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3287ec88df50429a934ebc6cf14606215e27b987"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.33+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "0d621a4beb5e48d195f907c3c5b0bea285d9ff9d"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.13+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.RationalRoots]]
git-tree-sha1 = "e5f5db699187a4810fda9181b34250deeedafd81"
uuid = "308eb6b3-cc68-5ff3-9e97-c3c4da4fa681"
version = "0.2.1"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "770240df9a3b8888065046948f7a09b4e0f997d5"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.Strided]]
deps = ["LinearAlgebra", "StridedViews", "TupleTools"]
git-tree-sha1 = "8c4f33a88bcd7dfee25ef0e59d724781ccd96b35"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "2.6.1"

    [deps.Strided.extensions]
    StridedAMDGPUExt = "AMDGPU"
    StridedGPUArraysExt = "GPUArrays"
    StridedcuBLASExt = "cuBLAS"

    [deps.Strided.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    cuBLAS = "182d3088-87b7-4494-8cad-fc6afaa545bc"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "21dc3942c478661f72c527ff5d67baa98e555372"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.5.2"

    [deps.StridedViews.extensions]
    StridedViewsAMDGPUExt = "AMDGPU"
    StridedViewsAdaptExt = "Adapt"
    StridedViewsCUDACoreExt = "CUDACore"
    StridedViewsJLArraysExt = "JLArrays"
    StridedViewsPtrArraysExt = "PtrArrays"

    [deps.StridedViews.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    PtrArrays = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TensorKit]]
deps = ["Adapt", "Dictionaries", "LRUCache", "LinearAlgebra", "MatrixAlgebraKit", "OhMyThreads", "Printf", "Random", "ScopedValues", "Strided", "TensorKitSectors", "TensorOperations", "TupleTools", "VectorInterface"]
git-tree-sha1 = "025081058ed953b53aeea9bb8bdbdf241a1fa54f"
uuid = "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"
version = "0.17.0"

    [deps.TensorKit.extensions]
    TensorKitAMDGPUExt = "AMDGPU"
    TensorKitCUDAExt = ["CUDA", "cuTENSOR"]
    TensorKitChainRulesCoreExt = "ChainRulesCore"
    TensorKitFiniteDifferencesExt = "FiniteDifferences"
    TensorKitMooncakeExt = "Mooncake"

    [deps.TensorKit.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"

[[deps.TensorKitSectors]]
deps = ["HalfIntegers", "LinearAlgebra", "TensorOperations", "WignerSymbols"]
git-tree-sha1 = "9f536263310aae90df175337d586a1655104aae3"
uuid = "13a9c161-d5da-41f0-bcbd-e1a08ae0647f"
version = "0.3.9"

[[deps.TensorOperations]]
deps = ["LRUCache", "LinearAlgebra", "PackageExtensionCompat", "PrecompileTools", "Preferences", "PtrArrays", "Strided", "StridedViews", "TupleTools", "VectorInterface"]
git-tree-sha1 = "c6153e90cf75256cb8b0ae451f0f7c0695dadd8d"
uuid = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
version = "5.6.2"

    [deps.TensorOperations.extensions]
    TensorOperationsAMDGPUExt = "AMDGPU"
    TensorOperationsBumperExt = "Bumper"
    TensorOperationsCUDACoreExt = "CUDACore"
    TensorOperationsChainRulesCoreExt = "ChainRulesCore"
    TensorOperationsEnzymeExt = "Enzyme"
    TensorOperationsJLArraysExt = "JLArrays"
    TensorOperationsMooncakeExt = "Mooncake"
    TensorOperationscuTENSORExt = "cuTENSOR"

    [deps.TensorOperations.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    Bumper = "8ce10254-0962-460f-a3d8-1f77fea1446e"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.VectorInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9166406dedd38c111a6574e9814be83d267f8aec"
uuid = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
version = "0.5.0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WignerSymbols]]
deps = ["HalfIntegers", "LRUCache", "Primes", "RationalRoots"]
git-tree-sha1 = "960e5f708871c1d9a28a7f1dbcaf4e0ee34ee960"
uuid = "9f57e263-0b3d-5e2e-b1be-24f2bb48858b"
version = "2.0.0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╟─94e4d9d6-a9a2-439b-ac84-c21aabbcd386
# ╟─1ea2749b-929d-4337-a018-2dafa6cbcf78
# ╠═b2b11530-a563-4e8f-b54c-4a9c419a4725
# ╟─7dd5af3d-93a0-4bc7-84d1-1833682f18cb
# ╟─f8effa9e-f7dc-4214-84fd-24719ef6892f
# ╠═1cb13598-8c18-4050-97a7-8b44318131aa
# ╠═8d57fea4-546c-4d21-b7e5-db0f27403e03
# ╠═687af255-f01b-4304-b470-0f61405b51d4
# ╠═a9f3f14a-54b1-4acf-853a-8c1d03e3259f
# ╠═9a9d0dce-838a-418f-9183-ff8eda820d97
# ╟─d56aea98-cbfe-4102-8efa-e559b1a524e2
# ╠═0eeee97a-c9cf-4604-b176-8016db3b68b9
# ╟─9703d795-e5c8-4b42-b7b6-c2ae32afddd4
# ╠═f3e8c698-5fa8-4188-84bf-34094a9bafc2
# ╟─fbb2f571-a2c5-4932-a317-06f9127ec681
# ╠═4e1a352b-e7ca-47bf-8e00-942db1e6f959
# ╠═dd0574dd-974a-4a5c-a5b9-a4d4b6043071
# ╟─cb04afcc-0a7e-4a5e-8f98-1d7db5996e39
# ╠═eac4e3b9-bb1d-4c91-b73a-989ce3ff2b56
# ╟─26bb9675-41b8-4b18-bfc9-e3fb28e9bd04
# ╠═ef03ea9a-888b-41ab-a662-3526739c377e
# ╠═45c52714-2a30-4603-a3ff-a551eddc54d6
# ╟─c133d327-cf58-42e0-9295-4b9c8e5fb4ab
# ╠═92553163-c6e5-4633-b81e-97c212003981
# ╟─ffa4defc-b6ef-4484-9649-53e18ce144d3
# ╟─0a4a016e-0616-4b80-95c9-638ab8f509b1
# ╠═98539830-cdfe-41d3-875c-a0525d2b73fd
# ╟─d6a6d30c-8333-4984-9a73-74f75c1ed1ab
# ╠═1bafd348-959c-4c6b-9266-cf1dd33aab7b
# ╠═cac61c91-b7c4-495b-b192-f65c029f851f
# ╟─77a9ae86-ac4c-4a93-ba34-9b692440f16d
# ╟─d840d4f4-212f-4ca8-81c7-2b1cc740d820
# ╠═e51c1c8e-3509-4147-a296-812027f9fcb8
# ╟─86b8c747-94e5-423b-9268-f3174d99c530
# ╠═2fad41c0-19ce-4770-b490-4fefde332f34
# ╟─618d38b8-7f5b-448b-a9d8-3788232068a3
# ╠═885414a7-b1fe-4edf-a0ba-7a5a51b3983f
# ╟─858ae02e-f54e-4498-a649-2e692fd2cf1f
# ╠═631e2384-66d3-43ed-80ff-4d9b1ee28a31
# ╟─b50d37ce-fefa-4934-a5f0-8786a4da1703
# ╠═89d563ea-8929-4ab0-9d15-ec59c0f90fe5
# ╟─71ce2788-1a86-4ab4-81e5-08f16caeb53b
# ╟─9524e71a-edbf-4118-84bb-0209698dcfe6
# ╟─5f58e0b3-e941-4694-98ec-0cb76b93ca13
# ╠═f229f3c8-ec88-4b59-96d6-9693a1d3c839
# ╠═12d8f0e6-3cbe-4543-90b3-491561c3010f
# ╟─7c0e0c02-893e-4e80-9741-6869a7c826ed
# ╟─23c24e86-25ac-4b94-a4da-94773e491c96
# ╠═c522c436-266e-4c82-8f30-93ddfc3ba1db
# ╟─3cd0da64-769e-42f9-8b37-20da133d42aa
# ╠═e1f2adf3-df9b-4094-9cad-de71cabf6e3e
# ╟─5c0561e9-1747-4069-8c10-72d26b7c2e9b
# ╟─1a310857-f3fc-4547-b0c0-c411af51e630
# ╟─5341c0e2-eb9f-4d14-874a-98c26742cecb
# ╠═52ef36df-f215-4ded-98f8-f9b68b9b0a91
# ╠═67d8e3ee-21f3-4f94-af11-5061fd24c77c
# ╟─8f87cda8-e656-4fab-8c1b-305126530def
# ╟─f3265508-02cd-4a53-8648-941ddd2bf5cc
# ╟─d52cf845-a2f1-45db-b6c6-eba149df00cb
# ╠═939dab51-20f9-4d76-9fac-4ce94f836ae2
# ╟─069866ec-8ca1-49d1-bf9e-c98652964529
# ╟─30f71f8b-b0ee-41fe-87be-0d15a9d1d2e9
# ╟─15ec7fc0-a50e-44e8-ba41-8cc918c8f5b5
# ╠═3583a901-f478-4ef3-a09b-fc6e6049fe14
# ╟─efab999d-2ccd-4002-83b4-2173de42e37f
# ╠═e204c1e0-318f-466e-89b4-823391c7cf58
# ╠═24bacdfd-ba40-4ac4-9a4b-586950be20e2
# ╟─61ff9b19-6ddc-472f-90b7-7222770932e9
# ╠═1cf78c09-abe4-4ef1-be3d-d29c0b117a1f
# ╟─d70f5db3-ade2-400e-9df8-83e2d2ca5169
# ╟─3f17f716-ee74-4666-a292-d962519a1a70
# ╟─42a1ace4-9d95-490c-82a1-06321afe1013
# ╟─084e8438-5641-4321-80f8-15d27b581040
# ╠═5887259a-11c6-497a-b25c-ac8af98fa41b
# ╟─ae357eba-748e-4cc5-a9de-906781af124f
# ╠═6f896e6f-bf9a-40aa-bde2-8fe3a462f9bb
# ╟─e41b227e-0d68-4eca-9055-3c479f39a22a
# ╟─7a106f3a-9ea1-4a25-a044-26633ef16604
# ╟─09ab54ba-6d99-4449-a1f6-06f8cee09b75
# ╟─a2b5fe58-2ed0-43b2-80c2-a8b044237969
# ╠═0a465a6e-c223-42fa-9eb1-48a198d2aeb0
# ╠═4f46a66a-5c29-4445-8b94-d7f72099f5a8
# ╟─c8a63111-d3aa-4c7f-98d3-6e031b9a628d
# ╠═14b9d4a2-1c80-4637-bd88-fa7eaabc43f7
# ╟─71441b76-d49d-4926-9e56-20fb57fc4dd4
# ╟─f3a38a7c-b29f-4c21-81cc-b8d7b114f9d4
# ╠═6eb05f47-4e80-491a-a5df-d6a90e5d200d
# ╠═27d4b849-26d6-4ece-9de5-0b2f35fc9470
# ╟─ada68578-e35d-448b-8ed1-35fbb61e1336
# ╠═9160eb93-bc34-4c9e-8abb-1f0baec13be1
# ╠═974b9dc4-ab44-48ce-accd-da102da5cbc6
# ╠═6cbe0a83-6645-4eb2-a323-8c89945c829c
# ╟─1fc05087-9a65-4113-bb1c-87b7023ec5cb
# ╟─db70cf96-c30d-44a7-9194-5ac7108376e1
# ╠═3245e811-8157-472d-9311-34654101b11e
# ╟─61ccf0a8-04b0-4e80-8d53-042acb1c3526
# ╟─6778a8a9-8ed2-4650-81ff-68b158ef05bf
# ╠═5e8ea935-fd55-4d1d-a9e3-4ce6c7e0194f
# ╠═93e32ca6-d730-4278-ad3d-2b032ca9c566
# ╠═be0b4b7b-e7b7-40e5-8b0e-1077c6a536b4
# ╠═0b1781be-9e9d-4eb1-b4da-cf289105c743
# ╟─158afe28-41b4-4834-a0e8-ec0505bf69b5
# ╟─c1c715d5-fc99-44d8-ad63-ca3702d4fcc6
# ╟─d05ebb75-976d-4919-bf4f-2d6a02eee150
# ╟─3b8c3cf1-4790-4775-8ca6-903ccf96d87e
# ╠═3321bb19-a246-4992-9991-799ca22b623e
# ╠═4f47cb0f-98c7-4013-9dbf-f76052cadffa
# ╠═180c9636-7400-4870-a7fb-e5665d6b1767
# ╟─66d6fbf6-8d72-4c5b-9a82-b0c0c17be1c2
# ╠═24a36817-886b-4555-8982-a5bc68c10bd5
# ╠═dbbe770e-4149-4bd1-bdea-dc52d13b3f11
# ╟─489242de-ad70-46c2-ad7d-f0fbd1e3f6f2
# ╟─ec0cad09-b01e-454a-bd14-bc01538b588b
# ╠═d1366be4-8877-4a10-993e-21bfdddbe22a
# ╠═8fe61599-26df-4e1b-a75b-009fc43486d2
# ╟─84ccbf05-bd50-4348-875f-87ae372a0e8b
# ╟─6eefd53d-c001-4696-bf83-039953c19c83
# ╠═eb8e42b4-e704-488c-80e8-d9afbc6bc162
# ╟─c7c85315-ff3c-4c69-b1f8-3ff0b077cb86
# ╠═30960ab5-4c3c-4cdb-a8c8-0335011b2864
# ╠═a310bf9c-8b9c-482f-9c4c-76551ea6bc27
# ╟─ece8ecc2-3a50-4c04-b314-50fa532f21f8
# ╟─ac718414-7df8-4d43-950f-bbe3a7831a6c
# ╠═7f696194-bcfc-4b46-8974-4cbebde2fd7c
# ╠═cad5d22b-4ab1-423b-84de-8e142568ddbc
# ╠═3b4561f9-f95d-4294-83ea-6ebf59ba1243
# ╠═31ee3fbe-f70f-4e94-b952-d455ec9564a6
# ╠═80847bbf-eb87-4d1e-a372-13ccb91131eb
# ╟─5de28869-1826-40b6-9258-7a9c4297e0bd
# ╟─bcf25e24-5646-4590-966f-7ca0b1ca41b8
# ╠═2dfb7a0e-e80e-4d63-93c7-8432e4c0c605
# ╟─86577720-cb91-4118-a561-31b082165e50
# ╟─f0284265-e6ab-4b32-89b4-2c67de4f87c1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
