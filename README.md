# AI4stats 

Short course on AI for statistical analysis.

## Keras installation guide.

We will be using `keras3` for this short course. You can find more details about `keras3` [here](https://keras3.posit.co/). In short the `keras3` package supersedes the older `keras` package. Previously, one would need to install `tensorflow` and `keras` to work with `keras`. My recent testing suggests that `keras3` simplifies the process a *lot*.

```{r}
install.packages('reticulate')
library(reticulate)
install.packages('keras3')
library(keras3)
install_keras()
set_random_seed(1)
```

The code chunk above should be sufficient for getting you up and running.

### Warning! 

Mixing `tensorflow` + `keras` (which is how we had to do things previously) with `keras3` is generally speaking going to be a bad time. If you come across old `keras` code that you need to run, the best option is to rewrite it in `keras3` - documentation is fairly good and almost all of the functionality translates over In general I would recommend not mixing the versions because of them sharing a lot of frontend names but with different backend funcitonality, leading to conflicts and errors. If you have to use the old `keras`, installing it in a virtual environment is probably your best bet. You can find a good tutorial [here](https://github.com/callumbarltrop/DeepGauge).

## Running the code during the short course.

For each of the codebooks, three versions have been provided:

1.  A `.Rmd` file. This is a markdown file that you should use only if you already have `keras3` installed. There are also certain cells you should not be running. If unclear, ask me.
2.  A `.html` file. This is compiled from the `.Rmd` file. I will be going over this for the most part. Best for following along without needing to run code.
3.  A `.ipynb` file. This is if you don't have `keras3` set up, but still want to take the code for a spin. This runs on Google Colab, with an `R` backend. The instructions ask you to set the backend to `R` manually, but I have found that once I saved the file, it retained that information.
