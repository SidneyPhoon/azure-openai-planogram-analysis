# Azure OpenAI Planogram Auditor

Using Azure OpenAI GPT 4o to audit retail planograms and product assortments. 
The AI auditor inspects each shelf of a cooler photo, determining if it complies with the planogram rules. The system identifies the presence of required SKUs and reports any discrepancies or competitor products.

## Features

- **Automated Auditing**: Uses AI to analyze images of cooler shelves.
- **Compliance Checking**: Evaluates whether the products on each shelf meet specified planogram rules.
- **Detailed Reporting**: Generates an audit table that summarizes compliance for each shelf.

## Requirements
* Azure OpenAI with GPT 4o enabled
* Open the [.env.sample](./.env.sample) file and replace the placeholders with your Azure OpenAI and Search credentials, save the file an name it `.env`.

## Payload Structure

The payload sent to the AI model consists of structured messages, including:

1. **System Message**: Describes the auditor's role and the product SKUs being assessed.
2. **User Task**: Specifies the auditing task, including the rules for each shelf and the expected output format.
3. **Planogram Rules**: Detailed requirements for each shelf, specifying the number of units for each SKU.
4. **Image Data**: The image of the cooler shelf is included as base64-encoded data.
