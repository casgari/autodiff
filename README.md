# Automatic Differentiation Package
-------

[![.github/workflows/test.yml](https://github.com/casgari/autodiff/blob/main/.github/workflows/test.yml/badge.svg)](https://github.com/casgari/autodiff/blob/main/.github/workflows/test.yml)
[![.github/workflows/coverage.yml](https://github.com/casgari/autodiff/blob/main/.github/workflows/coverage.yml/badge.svg)](https://github.com/casgari/autodiff/blob/main/.github/workflows/coverage.yml)

### Documentation
-------
[https://github.com/casgari/autodiff/blob/main/docs/documentation.ipynb](https://github.com/casgari/autodiff/blob/main/docs/documentation.ipynb)

### How to Install
-------
1. Navigate to desired directory and create virtual environment
```python
python -m venv test_env
```
2. Activate the environment 
```python
source test_env/bin/activate
```
3. Navigate inside test_env and install dependencies
```python
cd test_env
python -m pip install numpy
```
4. Install our package
```python
python -m pip install -i https://test.pypi.org/simple/ autodiff-team31==1.0.0
```
5. Write your code and import our package!
```python
>>>import autodiff as ad
>>>from autodiff.func import Func
>>>ad.sin(1)
0.8414709848078965
>>>f = lambda x: ad.sin(x)+x
>>>f1 = Func(f,1,1)
>>>f1(0)
0.0
>>>f1.jacobian(0)
1.0
# More examples in documentation
```
6. Deactivate the environment
```python
deactivate
``` 
### Broader Impact and Inclusivity

Our package was created to provide people with a user-friendly and readable introduction to automatic differentiation and its applications. We aimed to allow users to use our functions freely, handling input types and other specifications on the backend while maintaining functionality on the frontend. People will be able to efficiently perform forward and reverse mode automatic differentiation; additionally, this package supports function evaluations with specified parameters such as point and direction. We are certain that this software will provide consumers with unprecedented degrees of freedom given the input/output type support embedded within our functions; the computational possibilities are endless, ranging from basic elementary functions to the production of Jacobian matrices. This described versatility is deliberate, as we aimed to provide a multitude of services.

In an ideal world, the broader impact of our package ends there; however, it is essential to note the potential drawbacks of misuse and how that could impact our community. Forward and reverse mode automatic differentiation are a cornerstone of modern machine learning, specifically when training neural networks. These processes are used to fine tune input parameters (weights, biases, etc.) in hopes of producing a more accurate prediction. Today, there have been various cases of machine learning models propagating racism, sexism, and other forms of bigotry which are unacceptable to say the least. For example, in their 2018 research paper Gender Shades, Joy Buolamwini and Timnit Gebru touch upon a facial recognition model which performed better for lighter skinned individuals and worse for darker skinned individuals. A similar example comes from Nature Magazine, where researchers found that skin cancer classification models performed severely worse on darker skinned individuals; these models even optimized for lighter individuals as an attempt to maximize overall accuray. While these cases are not the norm in the field, they still require significant attention. It is our mission to provide our users with an unbiased, transparent tool, and we rebuke any implications that do not aim to do the same.

While we hope to make our package as inclusive as possible, there are evidently some "barriers to entry." Since most of our functionality is rooted in advanced mathematical topics, there are definitely features which some users will use ineffectively. We hoped to bridge that gap in the use of our comments and documentation, delineating inputs, outputs, and what our functions are performing; this use of text ensures that our consumers have a complete understanding of the intended and expected use. Additionally, each of our members comes from a different cultural, ethnic, and socioeconomic background. We aimed to make this product accessible and representative of all of our intended users. While we may lack gender diversity, our group deliberately made this product as gender-neutral as possible, emphasizing the validity and functionality of our solutions. Each addition to this project was carefully reviewed before finalizing its implementation. We ensured this by requiring that no one merge their own pull request so that the full team could provide extensive input before publishing it.

With all these thoughts in mind, we hope that you, our user, uses this package for its intended use. We anticipate our candidness sets an example for what we hope its impact to be.
