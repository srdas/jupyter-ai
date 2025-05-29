# The Finance Persona

This persona contains examples that showcase financial uses of agentic workflows. It uses APIs and tools from [financial datasets](https://www.financialdatasets.ai). It is based on a multi-agent system that uses the [Agno library](https://github.com/agno-agi/agno) under the hood. The persona is adept at responding to focused natural language instructions to perform various financial analyses and tasks such as collecting data, pricing securities, and forecasting. 

The Persona is packaged in a folder titled `financePersona` as containing the following files:

1. `finance_persona.py`: contains the main code for the persona. 
2. `fd.py`: This contains the `FinancialDatasetsTools` class with several functions and tools.
3. `requirements.txt`: for installing the various dependencies. 
4. This `README.md` file for documenting the finance persona. 
5. Various image files for the documentation.

The use of this persona requires an API key from Financial Datasets. You can register and find your account to get an API key. It has excellent tools for access to various quantitative and textual data. For details, see the [documentation](https://docs.financialdatasets.ai/introduction).

You will also update the TOML file `jupyter_ai_test/pyproject.toml` with the following line in the personas section:

```
finance-persona = "jupyter_ai_test.financePersona.finance_persona:FinancePersona"
```


