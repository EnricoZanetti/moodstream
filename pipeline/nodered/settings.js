module.exports = {
    flowFile: "flows.json",
    credentialSecret: false,
    uiPort: process.env.PORT || 1880,
    diagnostics: { enabled: true, ui: true },
    logging: {
        console: { level: "info", metrics: false, audit: false },
    },
};
